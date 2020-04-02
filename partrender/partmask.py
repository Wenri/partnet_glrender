from threading import Thread

from partrender.rendering import RenderObj
from tools.cfgreader import conf


class MaskObj(RenderObj):
    def __init__(self, start_id, auto_generate=False):
        self.bkt = None
        super(MaskObj, self).__init__(start_id, not auto_generate, conf.partmask_dir)

    def window_load(self, window):
        super(MaskObj, self).window_load(window)
        Thread(target=self, daemon=True).run()

    def __call__(self, *args, **kwargs):
        pass


def main(idx, autogen=True):
    show = MaskObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(0)
