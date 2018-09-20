using ImageClassification.Model;
using Microsoft.ML.Runtime.Api;

namespace ImageClassification.ImageData
{
    public class ImageNetPrediction
    {
        [ColumnName(ModelScorer.InceptionSettings.outputTensorName)]
        public float[] PredictedLabels;
    }
}
