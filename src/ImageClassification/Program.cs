using ImageClassification.Model;
using System;
using System.IO;
using System.Threading.Tasks;

namespace ImageClassification
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            // Running inside Visual Studio, $SolutionDir/assets is automatically passed as argument
            // If you execute from the console, pass as argument the location of the assets folder
            // Otherwise, it will search for assets in the executable's folder
            var assetsPath = args.Length > 0 ? args[0] : ModelHelpers.GetAssetsPath();

            var tagsTsv = Path.Combine(assetsPath, "inputs", "data", "tags.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "data");
            var inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            var labelsTxt = Path.Combine(assetsPath, "inputs", "inception", "imagenet_comp_graph_label_strings.txt");

            try
            {
                var modelEvaluator = new ModelScorer(tagsTsv, imagesFolder, inceptionPb, labelsTxt);
                modelEvaluator.Score();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception: {ex.Message}");
            }
            Console.ReadKey();
        }
    }
}
