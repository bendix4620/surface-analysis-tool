import unittest
import trimesh
import numpy as np
import numpy.testing as npt
from typing import Type

from sat import cache, data, jsonio
from sat.analysis import calculators, progress, registration
from sat.visual import geometries, plotter, slicing

progress.use_no_report()
SKIP_EXTERNAL_ALGORITHMS = True

class Test_Analysis_Calculators_SPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        act = trimesh.Trimesh(
            vertices=[
                [0, 0, -.5],
                [1, 0, -.5],
                [0, 1, -.5],
                [1, 1, -.5],
                [0, 0,  .5],
                [1, 0,  .5],
                [0, 1,  .5],
                [1, 1,  .5]],
            faces=[
                [0, 1, 2],  # lower planes
                [1, 3, 2],
                [0, 2, 5],  # diagonal planes
                [2, 7, 5],
                [4, 5, 6],  # upper planes
                [5, 7, 6]])
        nom = trimesh.Trimesh(
            vertices=[
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0]], 
            faces=[
                [0, 1, 2],
                [1, 3, 2]])
        corr = [0, 1, 0, 1, 0, 1]
        cls.calc = calculators.SPC(nom, act, corr)
        return super().setUpClass()

    def test_trimesh_properties(self):
        npt.assert_array_almost_equal(self.calc.nom.area_faces, [0.5, 0.5])
        npt.assert_almost_equal(self.calc.nom.area, 1.0)
        npt.assert_array_almost_equal(self.calc.act.area_faces**2, 
            [0.25, 0.25, 0.5, 0.5, 0.25, 0.25])
        npt.assert_almost_equal(self.calc.act.area, 3.414213562373095)
    
    def test_helping_parameters(self):
        npt.assert_array_almost_equal(self.calc.heights, 
            [[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], 
             [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5,  0.5,  0.5]])
        npt.assert_array_almost_equal(self.calc.absolute_triangle_heights, 
            [0.5, 0.5,  0.25,  0.25, 0.5, 0.5])
        npt.assert_array_almost_equal(self.calc.proj_act_tri_areas_signed, 
            [0.5, 0.5, -0.5,  -0.5,  0.5, 0.5])
        npt.assert_array_almost_equal(self.calc.proj_act_tri_areas, 
            [0.5, 0.5,  0.5,   0.5,  0.5, 0.5])
        npt.assert_almost_equal(self.calc.shadow_area, 1.0)
    
    def test_parameters_exist(self):
        cls = type(self.calc)
        for fname, dname in cls.params.items():
            self.assertTrue(hasattr(cls, fname), 
                msg=f"Attribute {fname} is missing in {cls.__name__}")
    
    def test_parameters(self):
        npt.assert_almost_equal(self.calc.Sa, 0.4166666666666667)
        npt.assert_almost_equal(self.calc.Sp, 0.5)
        npt.assert_almost_equal(self.calc.Sv, 0.5)
        npt.assert_almost_equal(self.calc.Sz, 1.0)
        npt.assert_almost_equal(self.calc.Sdr, 0.1380711874576983)
        npt.assert_almost_equal(self.calc.Sdr_p, 2.414213562373095)
        npt.assert_almost_equal(self.calc.Srf, 1.0)
        npt.assert_almost_equal(self.calc.Srr, 0.4142135623730951)


class Test_Analysis_Calculators_VPC(unittest.TestCase):
    def test_parameters_exist(self):
        cls = calculators.VPC
        for fname, dname in cls.params.items():
            self.assertTrue(hasattr(cls, fname), 
                msg=f"Attribute {fname} is missing in {cls.__name__}")

    def test_parameters(self):
        r = 3
        act = trimesh.creation.icosphere(radius=r, subdivisions=3)
        act.remove_unreferenced_vertices()
        calc = calculators.VPC(act=act)
        npt.assert_allclose(calc.Curv, 1/r, rtol=0.05)


class Test_Analysis_Registration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        a, b, c, d, e, h, g = 1, 3, 5, 1, 1, 2, 1
        cls.mesh = trimesh.Trimesh(
            vertices=[
                [0,   0,   0],  # 0
                [0,   a,   0],  # 1
                [0,   a, b-d],  # 2
                [0,   c, b-d],  # 3
                [0,   c,   b],  # 4
                [0,   0,   b],  # 5
                [g,   0,   0],  # 6
                [g,   a,   0],  # 7
                [g,   a, b-d],  # 8
                [g, c-e, b-d],  # 9
                [g, c-e,   b],  # 10
                [g,   0,   b],  # 11
                [h, c-e, b-d],  # 12
                [h,   c, b-d],  # 13
                [h,   c,   b],  # 14
                [h, c-e,   b]], # 15
            faces=[
                [ 0,  2,  1], # bottom
                [ 0,  5,  2],
                [ 2,  5,  4],
                [ 2,  4,  3],    
                [ 6,  7,  8], # mid
                [ 6,  8, 11],
                [ 8, 10, 11],
                [ 8,  9, 10],
                [12, 13, 14], # top
                [12, 14, 15],
                [ 0,  1,  6], # sides
                [ 1,  7,  6],
                [ 1,  2,  7],
                [ 2,  8,  7],
                [ 3,  8,  2],
                [ 3,  9,  8],
                [ 3,  4, 13],
                [ 4, 14, 13],
                [ 4, 15, 14],
                [ 4, 10, 15],
                [ 4, 11, 10],
                [ 4,  5, 11],
                [ 0, 11,  5],
                [ 0,  6, 11],
                [ 9, 12, 10],
                [10, 12, 15]])
        t1, t2, t3 = 2*np.pi*np.random.rand(3)
        Rx = np.array([[1,          0,           0],
                       [0,          np.cos(t1), -np.sin(t1)],
                       [0,          np.sin(t1),  np.cos(t1)]])
        Ry = np.array([[np.cos(t2), 0,           np.sin(t2)],
                       [0,          1,           0],
                       [-np.sin(t2), 0,           np.cos(t2)]])
        Rz = np.array([[np.cos(t3), -np.sin(t3), 0],
                       [np.sin(t3),  np.cos(t3), 0],
                       [0,           0,          1]])
        cls.T = np.eye(4)
        cls.T[:3, :3] = Rx @ Ry @ Rz
        cls.T[:3,  3] = 3*np.random.rand(3)
        return super().setUpClass()
    
    def test_inverse_transformation(self):
        H = registration.invT(registration.invT(self.T))
        npt.assert_array_almost_equal(self.T, H)

    @unittest.skipIf(SKIP_EXTERNAL_ALGORITHMS, "Trusting package authors")
    def test_align(self):
        nom = self.mesh.copy()
        act = self.mesh.copy()
        act.apply_transform(self.T)
        H = registration.align(nom, act)
        nom.apply_transform(H)

        distances = np.linalg.norm(nom.vertices - act.vertices, axis=1)
        npt.assert_allclose(distances, 0, rtol=0.01, atol=1e-12)
    
    def test_assign(self):
        nom = self.mesh.copy()
        act = self.mesh.copy()

        corr, dist = registration.assign(nom, act)
        npt.assert_array_equal(corr, np.arange(len(act.faces)))
        npt.assert_allclose(dist, 0, rtol=0.01, atol=1e-12)


class Test_Visual_Geometries(unittest.TestCase):
    def convextest(self, Body):
        """Standard test for convex bodies"""
        a = 1.1#1 + np.random.rand(1)[0]*3
        Tl = np.diag([a, a, a, 1])
        Ts = np.diag([1/a, 1/a, 1/a, 1])
        b = Body(center=[0, 0, 0])

        # larger mesh
        m = b.to_mesh()
        m.apply_transform(Tl)
        larger = b.contains(m.vertices)

        # smaller mesh
        m = b.to_mesh()
        m.apply_transform(Ts)
        smaller = b.contains(m.vertices)

        return np.all(smaller) and not np.any(larger)

    def test_Box_contains(self):
        self.assertTrue(self.convextest(geometries.Box))

    def test_Sphere_contains(self):
        self.assertTrue(self.convextest(geometries.Sphere))

    def test_TrianglePrism_contains(self):
        self.assertTrue(self.convextest(geometries.TrianglePrism))


class Test_Cache(unittest.TestCase):
    def test_compatible(self):
        class C:
            def __init__(self):
                self.cache = cache.Cache()
                foo = self.cache.newEntry(key="foo", name="FOO")
                bar = self.cache.newEntry(key="bar", name="BAR", src=[foo])
                return
            
            @cache.cached
            def foo(self):
                return 0
        
        c = C()
        self.assertFalse(c.cache.is_compatible(c, asbool=True))
        self.assertRaises(AttributeError, c.cache.is_compatible, 
                          c, asbool=False)
    
    def test_invalidation(self):
        class C:
            def __init__(self):
                self.cache = cache.Cache()
                foo = self.cache.newEntry(key="foo", name="FOO")
                bar = self.cache.newEntry(key="bar", name="BAR", src=[foo])
                return
            
            @cache.cached
            def foo(self):
                return "foo"
            
            @foo.setter
            def foo(self, val):
                self.cache["foo"] = val
            
            @cache.cached
            def bar(self):
                return str(self.foo) + "bar"
        
        c = C()
        self.assertTrue(c.cache.is_compatible(c, asbool=True))
        self.assertTrue(c.bar == "foobar")
        self.assertTrue(c.cache["bar"] == "foobar")
        c.foo = 1
        self.assertTrue(c.cache.getEntry("bar").isempty())
        
class Test_Data(unittest.TestCase):
    def assertEMPTY(self, obj, key, msg=None):
        self.assertTrue(obj.cache[key] is cache.EMPTY, msg)
    def assertFILLED(self, obj, key, msg):
        self.assertFalse(obj.cache[key] is cache.EMPTY, msg)

    def test_initial_cache(self):
        d = data.Data()
        
        # initial
        for key in d.cache.keys():
            if key == "nom" or key == "act":
                self.assertFILLED(d, key, 
                    f"Cache for Data.{key} should initially not be empty")
                continue
            self.assertEMPTY(d, key,
                f"Cache for Data.{key} should initially be empty")
        
        for key in d.nom.cache.keys():
            self.assertEMPTY(d.nom, key,
                f"Cache for Data.nom.{key} should initially be empty")

        for key in d.act.cache.keys():
            self.assertEMPTY(d.act, key,
                f"Cache for Data.act.{key} should initially be empty")

    def test_cascade_mesh(self):
        cascade = ("Entry source", [
            ("Entry mesh", [
                ("Entry transformation", []), 
                ("Entry subdivisions", [])
                ])
            ])
        m = data.Mesh(None)
        self.assertCountEqual(m.cache.getEntry("source").cascade(), cascade)
    
    def test_cascade_data(self):
        cascade = ("Cache Cache", [
            ("Entry corresponding faces", [
                ("Entry distance", []),
                ("Entry surface parameters", [])
                ])
            ])
        d = data.Data()
        self.assertCountEqual(d.nom.cache.cascade(), cascade)

        cascade = ("Cache Cache", [
            ("Entry corresponding faces", [
                ("Entry distance", []),
                ("Entry surface parameters", [])
                ]),
            ("Entry vertex parameters", [])
            ])
        d = data.Data()
        self.assertCountEqual(d.act.cache.cascade(), cascade)


class Test_JSONIO(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()