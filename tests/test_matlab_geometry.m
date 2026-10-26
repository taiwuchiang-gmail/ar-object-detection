function tests = test_matlab_geometry
%TEST_MATLAB_GEOMETRY Check empty renders and coordinate conversion.
tests = functiontests(localfunctions);
end
function testFullFrame(testCase)
verifyEqual(testCase,ar.yoloBox([0.5 0.5 640 480],[480 640]), ...
    [0.5 0.5 1 1],AbsTol=1e-12);
end
function testSmallBox(testCase)
verifyEqual(testCase,ar.yoloBox([10.5 20.5 20 10],[100 100]), ...
    [0.2 0.25 0.2 0.1],AbsTol=1e-12);
end
function testEmptyFrame(testCase)
verifyEmpty(testCase,ar.foregroundBox(zeros(32,32,3,'uint8'),1,2,0));
end
