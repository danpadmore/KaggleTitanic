using KaggleTitanic.Model;
using Microsoft.ML.Data;

namespace KaggleTitanic.AutoMl
{
    public class SurvivalPrediction : TestPassenger
    {
        [ColumnName("PredictedLabel")]
        public bool Survived { get; set; }

        public float Score { get; set; }
    }
}
