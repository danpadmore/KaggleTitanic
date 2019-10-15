using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static System.Console;

namespace KaggleTitanic
{
    class Program
    {
        private const string TrainedModelFilePath = "trained_model.zip";

        static void Main(string[] args)
        {
            try
            {
                WriteLine("Begin");

                if (ShouldTrain())
                {
                    Train();
                }

                var mlContext = new MLContext();
                var data = mlContext.Data.LoadFromTextFile<Passenger>(@"data\test.csv",
                    separatorChar: ',', hasHeader: true, allowQuoting: true);

                var predictionPipeline = mlContext.Model.Load(TrainedModelFilePath, out DataViewSchema predictionPipelineSchema);

                var predictions = predictionPipeline.Transform(data);
                var survivalPredictions = mlContext.Data.CreateEnumerable<SurvivalPrediction>(predictions, reuseRowObject: true);

                File.WriteAllLines("submission.csv", 
                    new List<string> {  "PassengerId,Survived" }
                    .Concat(survivalPredictions.Select(p => $"{p.PassengerId},{(p.Survived ? 1 : 0)}")));
            }
            catch (Exception ex)
            {
                WriteLine(ex);
            }
            finally
            {
                WriteLine("End");
                ReadLine();
            }
        }

        private static void Train()
        {
            var mlContext = new MLContext();

            var data = mlContext.Data.LoadFromTextFile<TrainingPassenger>(@"data\train.csv",
                separatorChar: ',', hasHeader: true, allowQuoting: true);
            var splittedData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var dataPipeline = mlContext.Transforms.Categorical.OneHotEncoding("SexEncoded", inputColumnName: nameof(TrainingPassenger.Sex))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("CabinEncoded", inputColumnName: nameof(TrainingPassenger.Cabin)))
                .Append(mlContext.Transforms.Text.FeaturizeText("NameFeaturized", nameof(TrainingPassenger.Name)))
                .Append(mlContext.Transforms.Concatenate("Features",
                    "NameFeaturized", nameof(TrainingPassenger.Pclass), "SexEncoded", nameof(TrainingPassenger.Age), nameof(TrainingPassenger.SibSp), nameof(TrainingPassenger.Parch), "CabinEncoded"));

            var trainer = mlContext.BinaryClassification.Trainers
                .SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(splittedData.TrainSet);

            var predictions = trainedModel.Transform(splittedData.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions,
                labelColumnName: "Label", scoreColumnName: "Score");

            PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
            mlContext.Model.Save(trainedModel, splittedData.TrainSet.Schema, TrainedModelFilePath);
        }

        public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            WriteLine($"************************************************************");
            WriteLine($"*       Metrics for {name} binary classification model      ");
            WriteLine($"*-----------------------------------------------------------");
            WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            WriteLine($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
            WriteLine($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
            WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            WriteLine($"************************************************************");
        }   

        private static bool ShouldTrain()
        {
            if (!File.Exists(TrainedModelFilePath))
            {
                return true;
            }

            WriteLine("Existing model found, delete and train new model? (Y)es or any key to skip");
            return (ReadKey().Key == ConsoleKey.Y);
        }
    }
}
