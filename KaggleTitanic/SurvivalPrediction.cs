using Microsoft.ML.Data;

namespace KaggleTitanic
{
    public class SurvivalPrediction : Passenger
    {
        [ColumnName("PredictedLabel")]
        public bool Survived { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }

        public override string ToString()
        {
            var result = Survived ? "survived" : "did not survive";

            return $"{Name} ({Age}/{Sex}/{SibSp}/{Parch}/{Pclass}/{Cabin}) {result} with a probability of {Probability} ({Score})";
        }
    }
}
