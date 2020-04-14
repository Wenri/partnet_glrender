import os
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
        im_id = conf.dblist[self.imageid]
        try:
            with open(os.path.join(conf.partmask_dir, im_id, 'render-CLSNAME.txt'),
                      mode='w') as f:
                for idx, mesh in enumerate(self.scene.mesh_list):
                    for material in mesh.materials:
                        conf_im_id, cls_name, file_name = conf.get_cls_from_mtlname(material.name)
                        assert conf_im_id == im_id
                        group_name = conf.find_group_name(cls_name)
                        print(conf_im_id, idx, group_name, cls_name, file_name, file=f)
            self.render_ack.wait()
            self.set_fast_switching()
        except RuntimeError:
            return


def main(idx, autogen=True):
    show = MaskObj(idx, autogen)
    show.show_obj()


if __name__ == '__main__':
    main(0)
