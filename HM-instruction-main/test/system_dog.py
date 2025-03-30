import utils.utils as utils


def test_file_now(file_path):
    try:
        with open(file_path, 'r') as file:
            print("文件可以正常打开。")
    except FileNotFoundError:
        print("文件未找到。")
    except PermissionError:
        print("没有权限打开文件。")
    except Exception as e:
        print(f"无法打开文件: {e}")

file_path = '../read.txt'

utils.test_file(file_path)
test_file_now(file_path)