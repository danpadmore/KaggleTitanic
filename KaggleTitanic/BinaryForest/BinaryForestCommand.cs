using KaggleTitanic.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static System.Console;

namespace KaggleTitanic.BinaryForest
{
    /// <summary>
    /// https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.treeextensions.fastforest?view=ml-dotnet#Microsoft_ML_TreeExtensions_FastForest_Microsoft_ML_BinaryClassificationCatalog_BinaryClassificationTrainers_System_String_System_String_System_String_System_Int32_System_Int32_System_Int32_
    /// </summary>
    public class BinaryForestCommand
    {
        private const string TrainedModelFilePath = "binary_forest_trained_model.zip";

        public static void Execute()
        {
            var mlContext = new MLContext();
            mlContext.ComponentCatalog.RegisterAssembly(typeof(PassengerTitleMappingFactory).Assembly);

            Train(mlContext);

            var data = mlContext.Data.LoadFromTextFile<TestPassenger>(@"data\test.csv",
                separatorChar: ',', hasHeader: true, allowQuoting: true);

            var predictionPipeline = mlContext.Model.Load(TrainedModelFilePath, out DataViewSchema predictionPipelineSchema);
            var predictions = predictionPipeline.Transform(data);
            var survivalPredictions = mlContext.Data.CreateEnumerable<SurvivalPrediction>(predictions, reuseRowObject: false);

            File.WriteAllLines("tree_submission.csv",
                new List<string> { "PassengerId,Survived" }
                .Concat(survivalPredictions.Select(p => $"{p.PassengerId},{(p.Survived ? 1 : 0)}")));
        }

        private static void Train(MLContext mlContext)
        {
            var data = mlContext.Data.LoadFromTextFile<TrainingPassenger>(@"data\train.csv",
                separatorChar: ',', hasHeader: true, allowQuoting: true);
            var splittedData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var dataPipeline = mlContext.Transforms.Categorical.OneHotEncoding("SexEncoded", inputColumnName: nameof(TrainingPassenger.Sex))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("EmbarkedEncoded", nameof(TrainingPassenger.Embarked)))
                .Append(mlContext.Transforms.CustomMapping(new PassengerTitleMappingFactory().GetMapping(), contractName: PassengerTitleMappingFactory.ContractName))
                .Append(mlContext.Transforms.Text.FeaturizeText("TitleFeaturized", inputColumnName: nameof(PassengerTitle.Title)))
                .Append(mlContext.Transforms.Text.FeaturizeText("LastNameFeaturized", "LastName"))
                .Append(mlContext.Transforms.Concatenate("Features",
                    "SexEncoded", nameof(TrainingPassenger.Pclass), nameof(TrainingPassenger.Age),
                    nameof(TrainingPassenger.SibSp),
                    "EmbarkedEncoded", "LastNameFeaturized", "TitleFeaturized"));

            var trainer = mlContext.BinaryClassification.Trainers.FastForest();
            var trainingPipeline = dataPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(splittedData.TrainSet);

            var predictions = trainedModel.Transform(splittedData.TestSet);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions,
                labelColumnName: "Label", scoreColumnName: "Score");

            PrintMetrics(metrics);
            mlContext.Model.Save(trainedModel, splittedData.TrainSet.Schema, TrainedModelFilePath);
        }

        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            WriteLine($"F1 Score: {metrics.F1Score:F2}");
            WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}
