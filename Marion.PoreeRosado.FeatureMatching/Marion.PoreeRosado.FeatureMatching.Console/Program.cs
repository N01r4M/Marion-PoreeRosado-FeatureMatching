using System.Text.Json;
using Marion.PoreeRosado.FeatureMatching;

var detectObjectInScenesResults = new ObjectDetection().DetectFromConsole(args);

foreach (var objectDetectionResult in detectObjectInScenesResults)
{
    System.Console.WriteLine($"Points: {JsonSerializer.Serialize(objectDetectionResult.Points)}");
}