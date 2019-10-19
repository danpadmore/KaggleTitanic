using KaggleTitanic.Model;
using Microsoft.ML.Data;

namespace KaggleTitanic.BinaryForest
{
    public class SurvivalPrediction : TestPassenger
    {
        [ColumnName("PredictedLabel")]
        public bool Survived { get; set; }
    }
}
