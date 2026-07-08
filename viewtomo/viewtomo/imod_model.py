# Minimal IMOD .mod binary reader/writer.
#
# Derived from PEETModelParser.py by Daven Vasishtan (TEMPy project,
# https://github.com/ccpem/TEMPy).  Only the read/write/iterate subset is
# retained here; distance/vector helpers that require TEMPy's Vector class are
# omitted.  Binary format specification:
#   http://bio3d.colorado.edu/imod/doc/binspec.html
#
# If PEETModelParser is importable (user has TEMPy installed) it is used
# instead of the bundled implementation, giving access to the full API.

try:
    from PEETModelParser import PEETmodel as ImodModel
except ImportError:
    from struct import unpack, pack
    from copy import deepcopy
    from math import ceil
    import numpy as np

    class ImodModel:
        """Read, filter, and write IMOD .mod files.

        Internal structure mirrors PEETModelParser so that code written against
        this class can be switched to PEETmodel by changing the import above.

        ``self.objs`` is a list of object dicts, each containing::

            {'ctrs': [{'points': np.ndarray shape (N,3), 'flags': int,
                       'time': int, 'surf': int, ...}, ...], ...}
        """

        def __init__(self, modelfile=''):
            if modelfile:
                with open(modelfile, 'rb') as f:
                    self._read(f)
            else:
                self._make_empty(1, [1])

        # ------------------------------------------------------------------
        # Public API
        # ------------------------------------------------------------------

        def get_all_contour_points(self):
            """Return a list of (N,3) ndarrays, one per non-empty contour."""
            out = []
            for o in self.objs:
                for c in o['ctrs']:
                    if len(c['points']) > 0:
                        out.append(np.array(c['points']))
            return out

        def write_model(self, outfile):
            all_pts = self.get_all_contour_points()
            if all_pts:
                flat = np.concatenate(all_pts, axis=0)
                self.max_values = (
                    int(ceil(flat[:, 0].max())),
                    int(ceil(flat[:, 1].max())),
                    int(ceil(flat[:, 2].max())),
                )
            self.no_of_obj = len(self.objs)
            hdr = pack(
                '>8s128s4i1I7i3f5i1f2i3f',
                self.id, self.header,
                self.max_values[0], self.max_values[1], self.max_values[2],
                self.no_of_obj, self.flags,
                self.drawmode, self.mousemode,
                self.blacklevel, self.whitelevel,
                self.offsets[0], self.offsets[1], self.offsets[2],
                self.scales[0], self.scales[1], self.scales[2],
                self.object, self.contour, self.point,
                self.res, self.thresh,
                self.pixsize, self.units, self.csum,
                self.alpha, self.beta, self.gamma,
            )
            body = b''.join(self._write_obj(o) for o in self.objs)
            footer = pack(f'>{len(self.footer)}s', self.footer)
            with open(outfile, 'wb') as f:
                f.write(hdr + body + footer)

        # ------------------------------------------------------------------
        # Private helpers
        # ------------------------------------------------------------------

        def _read(self, f):
            self.id          = unpack('>8s',   f.read(8))[0]
            self.header      = unpack('>128s',  f.read(128))[0]
            self.max_values  = unpack('>iii',   f.read(12))
            self.no_of_obj   = unpack('>i',     f.read(4))[0]
            self.flags       = unpack('>I',     f.read(4))[0]
            self.drawmode, self.mousemode   = unpack('>ii', f.read(8))
            self.blacklevel, self.whitelevel = unpack('>ii', f.read(8))
            self.offsets     = unpack('>iii',   f.read(12))
            self.scales      = unpack('>fff',   f.read(12))
            self.object, self.contour, self.point = unpack('>iii', f.read(12))
            self.res, self.thresh = unpack('>ii', f.read(8))
            self.pixsize     = unpack('>f',     f.read(4))[0]
            self.units       = unpack('>i',     f.read(4))[0]
            self.csum        = unpack('>i',     f.read(4))[0]
            self.alpha, self.beta, self.gamma = unpack('>fff', f.read(12))
            self.objs = [self._read_obj(f) for _ in range(self.no_of_obj)]
            self.footer = f.read()

        def _read_obj(self, f):
            tag = unpack('>4s', f.read(4))[0].decode('UTF-8')
            if tag != 'OBJT':
                raise TypeError(f'Expected OBJT chunk, got {tag!r}')
            obj = {}
            obj['id']         = tag
            obj['name']       = unpack('>64s', f.read(64))[0].decode('UTF-8')
            obj['extra']      = unpack('>64s', f.read(64))[0].decode('UTF-8')
            obj['no_of_ctrs'] = unpack('>i',   f.read(4))[0]
            obj['flags']      = unpack('>i',   f.read(4))[0]
            obj['axis'], obj['drawmode'] = unpack('>ii', f.read(8))
            obj['red'], obj['green'], obj['blue'] = unpack('>fff', f.read(12))
            obj['pdrawsize']  = unpack('>i',   f.read(4))[0]
            obj['symbols']    = unpack('>8B',  f.read(8))
            obj['meshsize']   = unpack('>i',   f.read(4))[0]
            obj['surfsize']   = unpack('>i',   f.read(4))[0]
            obj['ctrs'] = [self._read_ctr(f) for _ in range(obj['no_of_ctrs'])]
            return obj

        def _read_ctr(self, f):
            tag = unpack('>4s', f.read(4))[0].decode('UTF-8')
            # Skip SIZE chunks that can appear before CONT
            while tag == 'SIZE':
                f.read(4)  # skip SIZE value
                tag = unpack('>4s', f.read(4))[0]
                try:
                    tag = tag.decode('UTF-8')
                except UnicodeDecodeError:
                    pass
            if tag != 'CONT':
                raise TypeError(f'Expected CONT chunk, got {tag!r}')
            ctr = {'id': tag}
            ctr['psize'] = unpack('>i', f.read(4))[0]
            ctr['flags'] = unpack('>I', f.read(4))[0]
            ctr['time']  = unpack('>i', f.read(4))[0]
            ctr['surf']  = unpack('>i', f.read(4))[0]
            raw = unpack('>' + 'f' * 3 * ctr['psize'], f.read(12 * ctr['psize']))
            ctr['points'] = np.array(raw, dtype=float).reshape((ctr['psize'], 3))
            return ctr

        def _write_obj(self, obj):
            obj['no_of_ctrs'] = len(obj['ctrs'])
            blob = pack(
                '>132s4i3f1i8B2i',
                (obj['id'] + obj['name'] + obj['extra']).encode(),
                obj['no_of_ctrs'], obj['flags'], obj['axis'], obj['drawmode'],
                obj['red'], obj['green'], obj['blue'], obj['pdrawsize'],
                *obj['symbols'],
                obj['meshsize'], obj['surfsize'],
            )
            return blob + b''.join(self._write_ctr(c) for c in obj['ctrs'])

        def _write_ctr(self, ctr):
            pts = ctr['points']
            n = len(pts)
            blob = pack('>4s1i1I2i', ctr['id'].encode(), n, ctr['flags'],
                        ctr['time'], ctr['surf'])
            for p in pts:
                blob += pack('>3f', float(p[0]), float(p[1]), float(p[2]))
            return blob

        def _make_empty(self, no_of_objs, no_of_ctrs_list):
            self.id          = b'IMODV1.2'
            self.header      = ('ImodModel' + '\x00' * 119).encode()
            self.max_values  = (0, 0, 0)
            self.no_of_obj   = no_of_objs
            self.flags       = 61440
            self.drawmode, self.mousemode   = 1, 1
            self.blacklevel, self.whitelevel = 0, 255
            self.offsets     = (0, 0, 0)
            self.scales      = (1., 1., 1.)
            self.object, self.contour, self.point = 1, 1, 0
            self.res, self.thresh = 3, 128
            self.pixsize     = 1.
            self.units       = 0
            self.csum        = 0
            self.alpha, self.beta, self.gamma = 0., 0., 0.
            self.objs = []
            for n in no_of_ctrs_list:
                self.objs.append(self._empty_obj(n))
            self.footer = b'IEOF'

        def _empty_obj(self, no_of_ctrs):
            obj = {
                'id': 'OBJT', 'name': '\x00' * 64, 'extra': '\x00' * 64,
                'no_of_ctrs': 0, 'flags': 520, 'axis': 0, 'drawmode': 1,
                'red': 0., 'green': 1., 'blue': 0., 'pdrawsize': 0,
                'symbols': (1, 3, 1, 1, 0, 0, 0, 0), 'meshsize': 0, 'surfsize': 0,
                'ctrs': [self._empty_ctr() for _ in range(no_of_ctrs)],
            }
            return obj

        @staticmethod
        def _empty_ctr():
            return {'id': 'CONT', 'psize': 0, 'flags': 0, 'time': 0,
                    'surf': 0, 'points': np.array([], dtype=float)}
