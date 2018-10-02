using ImageClassification.Model;
using Microsoft.ML.Transforms.TensorFlow;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using static ImageClassification.Model.ConsoleHelpers;

namespace ImageClassification
{
    public class Program
    {
        static async Task Main(string[] args)
        {
            var assetsPath = ModelHelpers.GetAssetsPath(@"..\..\..\assets");

            var tagsTsv = Path.Combine(assetsPath, "inputs", "images", "tags.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "images");
            var inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb");
            var labelsTxt = Path.Combine(assetsPath, "inputs", "inception", "imagenet_comp_graph_label_strings.txt");

            var customInceptionPb = Path.Combine(assetsPath, "inputs", "inception_custom", "model_tf.pb");
            var customLabelsTxt = Path.Combine(assetsPath, "inputs", "inception_custom", "labels.txt");

            try
            {
                var modelEvaluator = new ModelScorer(tagsTsv, imagesFolder, inceptionPb, labelsTxt);
                modelEvaluator.Score();
                //var modelNodes = TensorFlowUtils.GetModelNodes(customInceptionPb).ToArray();
                //foreach (var item in modelNodes)
                //{
                //    Console.Out.WriteLine($"{item.Item1},{item.Item2}");
                //}
                //var modelEvaluatorCustom = new ModelScorerCustom(tagsTsv, imagesFolder, customInceptionPb, customLabelsTxt);
                //modelEvaluatorCustom.Score();
            }
            catch (Exception ex)
            {
                ConsoleWriteException(ex.Message);
            }

            ConsolePressAnyKey();
        }
    }
}
