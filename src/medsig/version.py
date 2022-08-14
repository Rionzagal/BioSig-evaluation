import os


def string():
    result: str = "unknown (git checkout)"
    try:
        with open(os.path.dirname(__file__) + "/VERSION", "r", encoding="utf-8") as file:
            version = file.read().strip()
            if version and "-" in version:
                v, i, s = version.split("-")
                version = f"{v}+{i}.git.{s}"
            result = version if version else result
    except Exception as ex:
        result = str(ex)
    return result
