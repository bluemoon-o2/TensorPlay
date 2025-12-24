import pickle
import os
import io
import zipfile
import json
import struct
import tensorplay as tp

DEFAULT_PROTOCOL = 2

def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True):
    """
    Saves an object to a disk file.
    Supports .tpm (TensorPlay Model) and .safetensors (Safetensors).
    """
    if isinstance(f, str):
        valid_extensions = ('.tpm', '.safetensors')
        if not f.endswith(valid_extensions):
            raise ValueError(f"Invalid file extension: '{f}'. Supported extensions are: {valid_extensions}")

        if f.endswith('.safetensors'):
            if not isinstance(obj, dict):
                 raise ValueError("Safetensors format only supports saving a dictionary of tensors.")
            return _save_safetensors(obj, f)
            
    if _use_new_zipfile_serialization:
        return _save_zip(obj, f, pickle_module, pickle_protocol)
    
    # Legacy direct pickle
    if isinstance(f, str):
        with open(f, "wb") as file:
            pickle_module.dump(obj, file, protocol=pickle_protocol)
    else:
        pickle_module.dump(obj, f, protocol=pickle_protocol)

def _load_onnx(filename):
    """
    Load ONNX model weights into a dictionary of TensorPlay tensors.
    """
    try:
        import onnx
        from onnx import numpy_helper
        import numpy as np
    except ImportError:
        raise ImportError("Loading ONNX files requires the 'onnx' library. Please install it via pip.")
        
    model = onnx.load(filename)
    weights = {}
    
    # Load initializers (weights)
    for initializer in model.graph.initializer:
        # Convert ONNX tensor to numpy
        # This handles raw data and data_type specifics
        np_array = numpy_helper.to_array(initializer)
        
        # Convert numpy to TensorPlay tensor
        # This uses the efficient from_dlpack/buffer path internally if available, or copy
        tp_tensor = tp.tensor(np_array)
        weights[initializer.name] = tp_tensor
        
    return weights

def _load_gguf(filename):
    """
    Load GGUF model weights into a dictionary of TensorPlay tensors.
    Uses C++ optimization for bulk reading of tensor data.
    """
    # Basic GGUF Header Structure (v3/v2 compatible-ish, focusing on v3)
    # Magic: 'GGUF' (4 bytes)
    # Version: uint32 (4 bytes)
    # Tensor Count: uint64 (8 bytes)
    # KV Count: uint64 (8 bytes)
    # ... KV pairs ...
    # ... Tensor Infos ...
    # ... Data ...
    
    with open(filename, 'rb') as f:
        # Read Magic
        magic = f.read(4)
        if magic != b'GGUF':
             raise ValueError(f"Invalid GGUF file: magic mismatch (got {magic})")
             
        # Read Version
        version = struct.unpack('<I', f.read(4))[0]
        if version not in (2, 3):
             # We can try to proceed but warn
             print(f"Warning: GGUF version {version} untest, only v2/v3 explicitly supported.")
             
        # Read Counts
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        
        # Helper to read strings
        def read_string(f):
            len_bytes = f.read(8)
            length = struct.unpack('<Q', len_bytes)[0]
            s = f.read(length).decode('utf-8')
            return s

        # Helper to read values based on type
        def read_value(f, type_id):
             # Simplified mapping for metadata values
             # 0: UINT8, 1: INT8, 2: UINT16, 3: INT16, 4: UINT32, 5: INT32
             # 6: FLOAT32, 7: BOOL, 8: STRING, 9: ARRAY, 10: UINT64, 11: INT64, 12: FLOAT64
             if type_id == 8: return read_string(f)
             elif type_id == 0: return struct.unpack('<B', f.read(1))[0]
             elif type_id == 1: return struct.unpack('<b', f.read(1))[0]
             elif type_id == 2: return struct.unpack('<H', f.read(2))[0]
             elif type_id == 3: return struct.unpack('<h', f.read(2))[0]
             elif type_id == 4: return struct.unpack('<I', f.read(4))[0]
             elif type_id == 5: return struct.unpack('<i', f.read(4))[0]
             elif type_id == 6: return struct.unpack('<f', f.read(4))[0]
             elif type_id == 7: return bool(struct.unpack('<B', f.read(1))[0])
             elif type_id == 10: return struct.unpack('<Q', f.read(8))[0]
             elif type_id == 11: return struct.unpack('<q', f.read(8))[0]
             elif type_id == 12: return struct.unpack('<d', f.read(8))[0]
             # Array support skipped for brevity/MVP
             elif type_id == 9:
                 arr_type = struct.unpack('<I', f.read(4))[0]
                 arr_len = struct.unpack('<Q', f.read(8))[0]
                 # Skip array content
                 # We need to know size of each element to skip
                 # This is getting complex for a simple loader. 
                 # Let's assume we don't need metadata arrays for weights loading for now
                 # But we must skip bytes to align file pointer.
                 # Implementation TODO: finish array skipping
                 for _ in range(arr_len):
                     read_value(f, arr_type)
                 return "<array>"
             else:
                 raise ValueError(f"Unknown GGUF value type: {type_id}")

        # Skip KV pairs
        for _ in range(kv_count):
            key = read_string(f)
            val_type = struct.unpack('<I', f.read(4))[0]
            val = read_value(f, val_type)
            # We could store metadata if needed
            
        # Parse Tensor Infos
        tensors_info = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            # GGUF stores dims in reverse order (like Fortran/PyTorch stride order logic sometimes)
            # but usually [n_dims] elements.
            # Check spec: "Dimensions of the tensor. The number of dimensions is `n_dims`."
            
            dtype_code = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            tensors_info.append({
                'name': name,
                'dims': dims, # Keep original order for now
                'dtype': dtype_code,
                'offset': offset
            })
            
        # Calculate base data offset
        # Alignment is usually 32 bytes or specified in metadata (general.alignment)
        # But we can infer it: current pos rounded up to alignment
        # Standard GGUF alignment is 32
        ALIGNMENT = 32
        current_pos = f.tell()
        remainder = current_pos % ALIGNMENT
        padding = (ALIGNMENT - remainder) if remainder != 0 else 0
        data_base_offset = current_pos + padding
        
    # Close file, now use C++ optimized loader
    weights = {}
    
    # GGUF DTypes Mapping
    # 0: F32, 1: F16, ... (Quantized types like Q4_0 etc not supported by base TensorPlay float/int types)
    # We will only support F32(0) and F16(1) for now.
    # If users need quantized, they need to dequantize or we add Q types to TensorPlay.
    
    for info in tensors_info:
        name = info['name']
        dims = info['dims']
        # GGUF dims are often stored reversed compared to NumPy? 
        # "Dimensions ... stored in reverse order" (ggml convention)
        # So we might need to reverse them back for PyTorch/TensorPlay shape
        shape = list(reversed(dims))
        
        dtype_code = info['dtype']
        offset = data_base_offset + info['offset']
        
        tp_dtype = None
        itemsize = 0
        
        if dtype_code == 0: # F32
            tp_dtype = tp.float32
            itemsize = 4
        elif dtype_code == 1: # F16
            # TensorPlay might not expose float16 easily yet in C++ binding as simple type
            # Assuming float32 for MVP or fail
            # Actually PyTorch supports half. TensorPlay _C might support it?
            # Looking at Tensor.cpp, DType has Float16?
            # Let's check DType enum. If not, we might fail.
            # Assuming we only support F32 for safety now.
             print(f"Skipping {name}: GGUF F16 not fully supported yet")
             continue
        else:
             print(f"Skipping {name}: Unsupported GGUF dtype {dtype_code}")
             continue
             
        numel = 1
        for d in shape: numel *= d
        nbytes = numel * itemsize
        
        # Call C++ optimized loader
        try:
             # Use static method on Tensor class
             tensor = tp.Tensor._load_file_segment(filename, offset, nbytes, shape, tp_dtype)
             weights[name] = tensor
        except Exception as e:
             print(f"Failed to load tensor {name}: {e}")
             
    return weights

def _save_zip(obj, f, pickle_module, pickle_protocol):
    # If f is a string, open it
    if isinstance(f, str):
        with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_STORED) as zf:
            _save_zip_internal(obj, zf, pickle_module, pickle_protocol)
    else:
        # f is a file-like object. We need to make sure it's seekable for ZipFile?
        # ZipFile 'w' mode doesn't strictly require seekable if we don't update.
        with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_STORED) as zf:
            _save_zip_internal(obj, zf, pickle_module, pickle_protocol)

def _save_zip_internal(obj, zf, pickle_module, pickle_protocol):
    # This dictionary maps unique IDs to serialized keys
    # We use id(tensor) to identify tensors during this save session
    saved_tensors = {}
    next_key = 0

    def persistent_id(obj):
        nonlocal next_key
        if isinstance(obj, tp.Tensor):
            # We identify tensor by its python object id for the scope of this serialization
            # Note: This doesn't handle views sharing storage efficiently yet, 
            # but it prevents infinite recursion if a tensor is referenced multiple times.
            obj_id = id(obj)
            if obj_id in saved_tensors:
                key = saved_tensors[obj_id]
            else:
                key = str(next_key)
                next_key += 1
                saved_tensors[obj_id] = key
                
                # Write tensor data to zip
                # Optimized: use buffer protocol to avoid copy if possible
                
                # Ensure CPU and contiguous
                # We can do this in python or assume C++ buffer protocol check
                t_save = obj
                if t_save.device.type != tp.DeviceType.CPU:
                    t_save = t_save.to("cpu")
                if not t_save.is_contiguous():
                    t_save = t_save.clone() # Make contiguous
                    
                # Write to archive/data/{key}
                with zf.open(f'archive/data/{key}', 'w') as df:
                    # zipfile.write supports buffer protocol objects
                    # use .detach().numpy() to get a zero-copy view that implements buffer protocol
                    # Note: t_save is already on CPU and contiguous from above checks
                    df.write(t_save.detach().numpy())
            
            # Return metadata tuple
            # ('tensorplay_tensor', key, shape, dtype, device, requires_grad)
            # We serialize device as string
            # Note: storing specific device index? obj.device.index?
            # tp.Device might have index. Let's assume generic 'cuda'/'cpu' for now 
            # or try to get string representation.
            # tp.Device objects usually print as "cpu" or "cuda:0".
            return ('tensorplay_tensor', key, list(obj.shape), str(obj.dtype), str(obj.device), obj.requires_grad)
        
        return None

    # Write pickle data
    # We write it to a buffer first, then to zip
    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    
    with zf.open('archive/data.pkl', 'w') as pf:
        pf.write(data_buf.getvalue())

def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """
    Loads an object saved with tensorplay.save() from a file.
    Supports .tpm, .safetensors, and .pth (via torch).

    Args:
        f: a file-like object (has to implement read, readline, tell, and seek), or a string containing a file name.
        map_location: a function, torch.device, string or a dict specifying how to remap storage locations
        pickle_module: module used for unpickling metadata and objects
        pickle_load_args: (optional) keyword arguments passed to pickle_module.load
    """
    if not isinstance(f, str):
        raise TypeError(f"Expected a string filename, got {type(f)}")
    if not os.path.exists(f):
        raise FileNotFoundError(f"No such file or directory: '{f}'")
    valid_extensions = ('.tpm', '.safetensors', '.pth', '.pt', '.bin', '.onnx', '.gguf')
    if not f.endswith(valid_extensions):
        raise ValueError(f"Invalid file extension: '{f}'. Supported extensions are: {valid_extensions}")

    if f.endswith('.safetensors'):
        return _load_safetensors(f)
    if f.endswith('.pth') or f.endswith('.pt') or f.endswith('.bin'):
        return _load_pth_compat(f, map_location)
    if f.endswith('.onnx'):
        return _load_onnx(f)
    if f.endswith('.gguf'):
        return _load_gguf(f)
    
    try:
        return _load_zip(f, map_location, pickle_module, **pickle_load_args)
    except (zipfile.BadZipFile, OSError):
        # Try legacy pickle
        if isinstance(f, str):
             with open(f, 'rb') as file:
                 return pickle_module.load(file, **pickle_load_args)
        else:
             f.seek(0)
             return pickle_module.load(f, **pickle_load_args)

def _load_zip(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    # If f is a string, open it
    with zipfile.ZipFile(f, 'r') as zf:
        # Read pickle data
        try:
            pickle_content = zf.read('archive/data.pkl')
        except KeyError:
            raise RuntimeError("Invalid tensorplay archive: 'archive/data.pkl' not found.")
            
        # Helper to map device
        def get_device(original_device_str):
            if map_location is None:
                return original_device_str
            
            if isinstance(map_location, str):
                return map_location
            elif isinstance(map_location, tp.Device):
                return str(map_location)
            elif isinstance(map_location, dict):
                return map_location.get(original_device_str, original_device_str)
            elif callable(map_location):
                return map_location(original_device_str)
            return original_device_str

        # Helper to convert dtype string back to tp.dtype
        # We need a mapping or eval.
        # Assuming we can access attributes of tp
        def get_dtype(dtype_str):
            # dtype_str example: "tensorplay.float32" or "float32"
            if "." in dtype_str:
                dtype_name = dtype_str.split(".")[-1]
            else:
                dtype_name = dtype_str
            
            if hasattr(tp, dtype_name):
                return getattr(tp, dtype_name)
            # Fallback
            return tp.float32

        def persistent_load(saved_id):
            tag, key, shape, dtype_str, device_str, requires_grad = saved_id
            if tag != 'tensorplay_tensor':
                raise RuntimeError(f"Unknown persistent id tag: {tag}")
            
            # Create tensor
            dtype = get_dtype(dtype_str)
            
            # Handle device
            target_device_str = get_device(device_str)
            
            # Check if we can load directly to device
            # _load_file_segment supports 'device' argument now (C++ updated)
            # If it's a zip stored file, we can read directly.
            
            # We need to find the file info for 'archive/data/{key}'
            try:
                # Direct load optimization only works if we have a real file on disk
                if isinstance(zf.filename, str) and os.path.exists(zf.filename):
                    info = zf.getinfo(f'archive/data/{key}')
                    if info.compress_type == zipfile.ZIP_STORED:
                        # Direct load optimized
                        # Offset calculation
                        # header_offset is the start of the local file header
                        # local header size = 30 + n + m
                        data_offset = info.header_offset + 30 + len(info.filename.encode('utf-8')) + len(info.extra)
                        
                        # Convert target_device_str to Device object if needed or pass string?
                        # _from_bytes doesn't take device, but _load_file_segment does (we just added it)
                        # Let's map target_device_str to tp.Device
                        if target_device_str == "cpu":
                            target_device = tp.Device(tp.DeviceType.CPU)
                        elif "cuda" in target_device_str:
                            # naive parsing
                            if ":" in target_device_str:
                                idx = int(target_device_str.split(":")[1])
                                target_device = tp.Device(tp.DeviceType.CUDA, idx)
                            else:
                                target_device = tp.Device(tp.DeviceType.CUDA, 0)
                        else:
                            target_device = tp.Device(tp.DeviceType.CPU) # Fallback
                            
                        # Calculate size
                        dummy = tp.empty((1,), dtype=dtype)
                        itemsize = dummy.itemsize()
                        nbytes = 1
                        for s in shape: nbytes *= s
                        nbytes *= itemsize
                        
                        t = tp.Tensor._load_file_segment(f.filename, data_offset, nbytes, shape, dtype, target_device)
                        
                        if requires_grad:
                            t.requires_grad = True
                        return t
                    
            except Exception as e:
                # Fallback to read + copy
                pass

            # Read data (slow path or compressed)
            data_bytes = zf.read(f'archive/data/{key}')
            
            # Optimized: use C++ _from_bytes (creates on CPU)
            t = tp.Tensor._from_bytes(data_bytes, shape, dtype)
            
            # Move to device if needed
            if target_device_str != "cpu" and str(t.device) != target_device_str:
                t = t.to(target_device_str)
                
            if requires_grad:
                t.requires_grad = True
                
            return t

        data_io = io.BytesIO(pickle_content)
        unpickler = pickle_module.Unpickler(data_io, **pickle_load_args)
        unpickler.persistent_load = persistent_load
        return unpickler.load()


# --- Safetensors Implementation ---

def _save_safetensors(tensors_dict, filename, metadata=None):
    """
    Save a dictionary of tensors to a safetensors file.
    Optimized to use C++ bulk write.
    """
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
        
    current_offset = 0
    tensor_list = []
    
    # Sort keys to ensure deterministic order
    keys = sorted(tensors_dict.keys())
    
    for key in keys:
        tensor = tensors_dict[key]
        if not isinstance(tensor, tp.Tensor):
             raise ValueError(f"Value for {key} is not a tensorplay.Tensor")
             
        # Map dtype
        dtype_str = _dtype_to_safetensors(tensor.dtype)
        
        # Calculate size (contiguous)
        nbytes = tensor.numel() * tensor.itemsize()
        
        header[key] = {
            "dtype": dtype_str,
            "shape": list(tensor.shape),
            "data_offsets": [current_offset, current_offset + nbytes]
        }
        
        current_offset += nbytes
        tensor_list.append(tensor)
        
    # Create header JSON
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    
    n_header = len(header_json)
    
    with open(filename, 'wb') as f:
        # Write N (u64)
        f.write(struct.pack('<Q', n_header))
        # Write Header
        f.write(header_json)
        # Data is appended by C++
        
    # Call C++ to append data
    tp.Tensor._save_file_segments(filename, tensor_list)

def _load_safetensors(filename):
    """
    Load a safetensors file.
    Optimized to use C++ bulk read.
    """
    with open(filename, 'rb') as f:
        # Read N
        n_header_bytes = f.read(8)
        if len(n_header_bytes) != 8:
            raise ValueError("Invalid safetensors file")
        n_header = struct.unpack('<Q', n_header_bytes)[0]
        
        # Read Header
        header_json = f.read(n_header)
        header = json.loads(header_json)
        
    # Prepare list for C++
    segments = []
    result = {}
    
    # Base offset for data = 8 + n_header
    base_offset = 8 + n_header
    
    for key, metadata in header.items():
        if key == "__metadata__":
            continue
            
        shape = metadata['shape']
        dtype_str = metadata['dtype']
        data_offsets = metadata['data_offsets']
        
        dtype = _dtype_from_safetensors(dtype_str)
        
        # Create empty tensor on CPU
        tensor = tp.empty(shape, dtype=dtype, device='cpu')
        
        start = data_offsets[0] + base_offset
        length = data_offsets[1] - data_offsets[0]
        
        segments.append((tensor, start, length))
        result[key] = tensor
        
    # Bulk load
    tp.Tensor._load_file_segments(filename, segments)
    
    return result

def _dtype_to_safetensors(dtype):
    s = str(dtype)
    if "float32" in s: return "F32"
    if "float64" in s: return "F64"
    if "int32" in s: return "I32"
    if "int64" in s: return "I64"
    if "int8" in s: return "I8"
    if "uint8" in s: return "U8"
    if "bool" in s: return "BOOL"
    raise ValueError(f"Unsupported dtype for safetensors: {dtype}")

def _dtype_from_safetensors(s):
    if s == "F32": return tp.float32
    if s == "F64": return tp.float64
    if s == "I32": return tp.int32
    if s == "I64": return tp.int64
    if s == "I8": return tp.int8
    if s == "U8": return tp.uint8
    if s == "BOOL": return tp.bool
    raise ValueError(f"Unsupported safetensors dtype: {s}")

# --- PTH Compatibility ---

def _load_pth_compat(filename, map_location=None):
    """
    Load PyTorch .pth/.pt/.bin files without requiring torch installed.
    Uses direct zip reading and pickle interception.
    """
    if not zipfile.is_zipfile(filename):
         # Try legacy pickle (not zip) - unlikely for modern files, but possible
         raise ValueError(f"File {filename} is not a zip file. Only standard PyTorch zip archives are supported natively.")

    with zipfile.ZipFile(filename, 'r') as zf:
        # 1. Find the pickle file (data.pkl)
        # PyTorch saves as archive/data.pkl usually
        pkl_name = None
        for n in zf.namelist():
            if n.endswith('data.pkl'):
                pkl_name = n
                break
                
        if not pkl_name:
             # Try finding ANY .pkl?
             candidates = [n for n in zf.namelist() if n.endswith('.pkl')]
             if candidates:
                 pkl_name = candidates[0]
             else:
                 raise ValueError("Could not find data.pkl in .pth file. Only standard PyTorch zip archives are supported.")
        
        base_dir = os.path.dirname(pkl_name)
        
        # 2. Setup Unpickler
        loaded_storages = {} # key -> Tensor (flat)
        
        class MockTorchUtils:
            @staticmethod
            def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                # storage is a TensorPlay tensor (1D)
                if storage is None: return None
                
                # Apply view/stride
                t = storage.as_strided(list(size), list(stride), storage_offset)
                if requires_grad:
                    t.requires_grad = True
                return t
                
            @staticmethod
            def _rebuild_parameter(data, requires_grad, backward_hooks):
                t = data
                if t is not None:
                    t.requires_grad = requires_grad
                return t

        class PthUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Intercept torch functions used for reconstruction
                if module == 'torch._utils':
                    if name == '_rebuild_tensor_v2': return MockTorchUtils._rebuild_tensor_v2
                    if name == '_rebuild_parameter': return MockTorchUtils._rebuild_parameter
                
                # Intercept Types
                if module.startswith('torch'):
                    if 'FloatStorage' in name: return 'FloatStorage'
                    if 'LongStorage' in name: return 'LongStorage'
                    if 'IntStorage' in name: return 'IntStorage'
                    if 'ShortStorage' in name: return 'ShortStorage'
                    if 'CharStorage' in name: return 'CharStorage'
                    if 'ByteStorage' in name: return 'ByteStorage'
                    if 'HalfStorage' in name: return 'HalfStorage'
                    if 'BoolStorage' in name: return 'BoolStorage'
                    if 'DoubleStorage' in name: return 'DoubleStorage'
                    if 'BFloat16Storage' in name: return 'BFloat16Storage'
                    if name == 'Tensor': return tp.Tensor 
                    
                    class Dummy: pass
                    return Dummy
                    
                return super().find_class(module, name)

            def persistent_load(self, saved_id):
                # saved_id: ('storage', storage_type, key, location, size)
                if isinstance(saved_id, tuple) and saved_id[0] == 'storage':
                    storage_type, key, location, size = saved_id[1:]
                    
                    if key in loaded_storages:
                        return loaded_storages[key]
                    
                    # Determine dtype
                    stype = str(storage_type)
                    if 'Float' in stype: dtype = tp.float32
                    elif 'Double' in stype: dtype = tp.float64
                    elif 'Long' in stype: dtype = tp.int64
                    elif 'Int' in stype: dtype = tp.int32
                    elif 'Short' in stype: dtype = tp.int16
                    elif 'Char' in stype: dtype = tp.int8
                    elif 'Byte' in stype: dtype = tp.uint8
                    elif 'Bool' in stype: dtype = tp.bool
                    elif 'Half' in stype: dtype = tp.float32 # Cast to f32
                    elif 'BFloat16' in stype: dtype = tp.float32 # Cast to f32
                    else: dtype = tp.float32
                    
                    # Read Data
                    storage_path = f"{base_dir}/data/{key}"
                    if base_dir == "": storage_path = f"data/{key}"
                    
                    # Try to find file
                    try:
                        info = zf.getinfo(storage_path)
                    except KeyError:
                        # Sometimes path is different?
                        try:
                             info = zf.getinfo(key) # Direct key?
                             storage_path = key
                        except KeyError:
                             # print(f"Warning: Storage {key} not found")
                             return None

                    # Use C++ bulk read if stored (uncompressed)
                    if info.compress_type == zipfile.ZIP_STORED:
                         # Calculate file offset
                         data_offset = info.header_offset + 30 + len(info.filename.encode('utf-8')) + len(info.extra)
                         
                         # Calculate size in bytes
                         dummy = tp.empty((1,), dtype=dtype)
                         itemsize = dummy.itemsize()
                         nbytes = size * itemsize
                         
                         t = tp.Tensor._load_file_segment(filename, data_offset, nbytes, [size], dtype)
                         loaded_storages[key] = t
                         return t
                    else:
                        # Fallback: read from zip (compressed)
                        data = zf.read(storage_path)
                        t = tp.Tensor._from_bytes(data, [size], dtype)
                        loaded_storages[key] = t
                        return t
                
                return None

        # Load
        with zf.open(pkl_name, 'r') as f:
             unpickler = PthUnpickler(f)
             return unpickler.load()