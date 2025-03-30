
def test_file(file_path):
    try:
        with open(file_path, 'r') as file:
            print("文件可以正常打开。")
    except FileNotFoundError:
        print("文件未找到。")
    except PermissionError:
        print("没有权限打开文件。")
    except Exception as e:
        print(f"无法打开文件: {e}")

