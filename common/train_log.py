from common.dataset import PATCH_PARAMS, result


def open_log(version, hparams):
    f = open(result(version, "result.log"), "w")

    def _section(title, d):
        f.write(f"[{title}]\n")
        for k, v in d.items():
            f.write(f"  {k} = {v}\n")
        f.write("\n")

    _section("dataset", PATCH_PARAMS)
    _section("hparams", hparams)
    f.flush()

    def log(msg):
        print(msg)
        f.write(msg + "\n")
        f.flush()

    return log
