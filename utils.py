import subprocess

from huggingface_hub.repository import _lfs_log_progress


def force_git_push(
    repo,
):
    """
    force a simple git push
    Blocking. Will return url to commit on remote
    repo.
    """
    command = "git push --force"

    try:
        with _lfs_log_progress():
            process = subprocess.Popen(
                command.split(),
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                encoding="utf-8",
                cwd=repo.local_dir,
            )

            stdout, stderr = process.communicate()
            return_code = process.poll()
            process.kill()

            if len(stderr):
                print(stderr)

            if return_code:
                raise subprocess.CalledProcessError(return_code, process.args, output=stdout, stderr=stderr)

    except subprocess.CalledProcessError as exc:
        raise EnvironmentError(exc.stderr)

    return repo.git_head_commit_url()
