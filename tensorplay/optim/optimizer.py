import tensorplay as tp
from collections import defaultdict

class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, tp.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in a set will change between runs. '
                            'Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, tp.Tensor):
                raise TypeError("optimizer can only optimize Tensors, but one of the params is " + str(type(param)))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            raise ValueError("optimizer contains a parameter group with duplicate parameters")

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
        
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def step(self, closure=None):
        raise NotImplementedError

    def state_dict(self):
        # Pack state and param_groups into a dictionary
        # We need to map parameters to ids because parameters are objects
        # In PyTorch, parameters are mapped to integer ids.
        
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = []
            for p in group['params']:
                if p not in param_mappings:
                    param_mappings[p] = len(param_mappings)
                packed['params'].append(param_mappings[p])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        
        packed_state = {}
        for p, s in self.state.items():
            if p in param_mappings:
                packed_state[param_mappings[p]] = s
        
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        # deepcopy, to be safe
        state_dict = state_dict.copy() # Shallow copy is enough for structure usually
        
        # Validate state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")

        # Update parameter groups
        id_map = {}
        for i, (group, saved_group) in enumerate(zip(groups, saved_groups)):
            for key, value in saved_group.items():
                if key != 'params':
                    group[key] = value
            
            for p, p_id in zip(group['params'], saved_group['params']):
                id_map[p_id] = p

        # Update state
        self.state = defaultdict(dict)
        for p_id, s in state_dict['state'].items():
            if p_id in id_map:
                self.state[id_map[p_id]] = s

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string
