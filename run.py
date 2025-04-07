import subprocess

# Gitの操作を実行 (例: リポジトリの状態を確認)
subprocess.run(["git", "status"])

# Pythonスクリプトを実行 (例: 任意のPythonスクリプトを実行)
subprocess.run(["python", "main.py"])

# さらにGitの操作を実行 (例: コミットを実行)
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "l1 visualize DBSCAN add"])
subprocess.run(["git", "push","origin","experiment"])
