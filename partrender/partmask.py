from threading import Thread

from partrender.rendering import RenderObj
from tools.cfgreader import conf


class MaskObj(RenderObj):
    def __init__(self, start_id, auto_generate=False):
        self.bkt = None
        super(MaskObj, self).__init__(start_id, not auto_generate, conf.partmask_dir)

    def window_load(self, window):
        super(MaskObj, self).window_load(window)
        Thread(target=self, daemon=True).start()

    def __call__(self, *args, **kwargs):
        try:
            for i in range(len(self.scene.mesh_list)):
                with self.set_render_name(str(i)):
                    self.sel_set = {i}
        except RuntimeError:
            return
        self.set_fast_switching()


def main(idx, autogen=True):
    show = MaskObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(0)
