using OpenCvSharp;

namespace Marion.PoreeRosado.FeatureMatching;

public class ObjectDetection
{
    private ObjectDetectionResult DetectObjectInScene(byte[] imageObjectData, byte[] imageSceneData)
    {
        using var imgobject = Mat.FromImageData(imageObjectData, ImreadModes.Color);
        using var imgScene = Mat.FromImageData(imageSceneData, ImreadModes.Color);
        using var orb = ORB.Create(10000);
        using var descriptors1 = new Mat();
        using var descriptors2 = new Mat();
        orb.DetectAndCompute(imgobject, null, out var keyPoints1, descriptors1);
        orb.DetectAndCompute(imgScene, null, out var keyPoints2, descriptors2);
        using var bf = new BFMatcher(NormTypes.Hamming, crossCheck: true);
        
        var matches = bf.Match(descriptors1, descriptors2);
        var goodMatches = matches
            .OrderBy(x => x.Distance)
            .Take(10)
            .ToArray();
        var srcPts = goodMatches.Select(m => keyPoints1[m.QueryIdx].Pt).Select(p => new Point2d(p.X, p.Y));
        var dstPts = goodMatches.Select(m => keyPoints2[m.TrainIdx].Pt).Select(p => new Point2d(p.X, p.Y));
        using var homography = Cv2.FindHomography(srcPts, dstPts, HomographyMethods.Ransac, 5, null);
        int h = imgobject.Height, w = imgobject.Width;
        var img2Bounds = new[]
        {
            new Point2d(0, 0),
            new Point2d(0, h-1),
            new Point2d(w-1, h-1),
            new Point2d(w-1, 0),
        };
        var img2BoundsTransformed = Cv2.PerspectiveTransform(img2Bounds, homography);
        using var view = imgScene.Clone();
        var drawingPoints = img2BoundsTransformed.Select(p => (Point) p).ToArray();
        Cv2.Polylines(view, new []{drawingPoints}, true, Scalar.Red, 3);
        // Uncomment to see the image result
        /*
        using (new Window("view", view))
        {
        Cv2.WaitKey();
        }
        */

        var imageResult = view.ToBytes(".png");
        return new ObjectDetectionResult()
        {
            ImageData = imageResult,
            Points = drawingPoints.Select(point => new ObjectDetectionPoint(){X=point.X, Y=point.Y} ).ToList()
        };
    }

    public async Task<IList<ObjectDetectionResult>> DetectObjectInScenesAsync(byte[] objectImageData, IList<byte[]> imagesSceneData)
    {
        IList<ObjectDetectionResult> results = new List<ObjectDetectionResult>();
        var tasks = new List<Task<ObjectDetectionResult>>();

        foreach (var image in imagesSceneData)
        {
            var task = Task.Run(() => DetectObjectInScene(objectImageData, image));
            tasks.Add(task);
        }

        Task.WaitAll(tasks.ToArray());

        foreach (var task in tasks)
        {
            results.Add(task.Result);
        }

        return results;
    }

    public IList<ObjectDetectionResult> DetectFromConsole(string[] args)
    {
        var images = new List<byte[]>();
        var results = new List<ObjectDetectionResult>();

        byte[] imageObject = File.ReadAllBytes(args[0]);

        foreach (var file in Directory.EnumerateFiles(args[1]))
        {
            var imageBytes = File.ReadAllBytes(file);
            images.Add(imageBytes);
        }

        foreach (var image in images)
        {
            var points = DetectObjectInScene(imageObject, image);
            results.Add(points);
        }

        return results;
    }
}
