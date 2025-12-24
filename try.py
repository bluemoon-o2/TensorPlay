import os
import shutil
import argparse


def replace_torch_with_tensorplay(file_path: str, backup: bool = True, new_file: bool = False) -> None:
    """
    将指定.py文件中的所有"torch"字符串替换为"tensorplay"

    Args:
        file_path: 目标.py文件的路径
        backup: 是否备份原文件（默认True，备份为原文件+.bak后缀）
        new_file: 是否生成新文件（默认False，覆盖原文件；True则生成原文件名+_modified.py）

    Raises:
        FileNotFoundError: 指定的文件不存在
        PermissionError: 无文件读写权限
        ValueError: 指定的文件不是.py文件
    """
    # 1. 校验文件是否为.py文件
    if not file_path.endswith(".py"):
        raise ValueError(f"错误：文件 {file_path} 不是.py文件，请指定有效的Python文件")

    # 2. 校验文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：文件 {file_path} 不存在")

    # 3. 备份原文件（如果开启备份）
    if backup:
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        print(f"已备份原文件至：{backup_path}")

    # 4. 读取文件内容
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except PermissionError:
        raise PermissionError(f"错误：无权限读取文件 {file_path}")
    except Exception as e:
        raise Exception(f"读取文件时发生错误：{str(e)}")

    # 5. 替换字符串（全局替换所有"torch"）
    modified_content = (content.replace("torch", "tensorplay").
                        replace("PyTorch", "TensorPlay").replace("TORCH", "TENSORPLAY"))

    # 6. 写入内容（覆盖原文件或生成新文件）
    if new_file:
        new_file_path = os.path.splitext(file_path)[0] + "_modified.py"
        write_path = new_file_path
    else:
        write_path = file_path

    try:
        with open(write_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
        print(f"替换完成！修改后的文件已保存至：{write_path}")
    except PermissionError:
        raise PermissionError(f"错误：无权限写入文件 {write_path}")
    except Exception as e:
        raise Exception(f"写入文件时发生错误：{str(e)}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="将Python文件中的'torch'替换为'tensorplay'")
    parser.add_argument("file_path", help="目标.py文件的路径（如：test.py）")
    parser.add_argument("--no-backup", action="store_false", dest="backup", help="不备份原文件（默认备份）")
    parser.add_argument("--new-file", action="store_true", help="生成新文件（不覆盖原文件，默认覆盖）")

    args = parser.parse_args()

    # 执行替换操作
    try:
        replace_torch_with_tensorplay(args.file_path, args.backup, args.new_file)
    except Exception as e:
        print(f"执行失败：{str(e)}")


if __name__ == "__main__":
    main()