using KaggleTitanic.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static System.Console;

namespace KaggleTitanic.AutoMl
{
    /// <summary>
    /// Based on ML.NET Model Builder tool
    /// </summary>
    public class FastForestCommand
    {
        private const string TrainedModelFilePath = "automl_fast_forest_trained_model.zip";

        public static void Execute()
        {
            var mlContext = new MLContext();

            Train(mlContext);

            var data = mlContext.Data.LoadFromTextFile<TestPassenger>(@"data\test.csv",
                separatorChar: ',', hasHeader: true, allowQuoting: true);

            var predictionPipeline = mlContext.Model.Load(TrainedModelFilePath, out DataViewSchema predictionPipelineSchema);
            var predictions = predictionPipeline.Transform(data);
            var survivalPredictions = mlContext.Data.CreateEnumerable<SurvivalPrediction>(predictions, reuseRowObject: false);

            File.WriteAllLines("automl_submission.csv",
                new List<string> { "PassengerId,Survived" }
                .Concat(survivalPredictions.Select(p => $"{p.PassengerId},{(p.Survived ? 1 : 0)}")));
        }

        private static void Train(MLContext mlContext)
        {
            var data = mlContext.Data.LoadFromTextFile<TrainingPassenger>(@"data\train.csv",
                separatorChar: ',', hasHeader: true, allowQuoting: true);
            var splittedData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var trainingPipeline = BuildTrainingPipeline(mlContext);
            var trainedModel = TrainModel(splittedData.TrainSet, trainingPipeline);

            Evaluate(mlContext, splittedData.TrainSet, trainedModel);

            mlContext.Model.Save(trainedModel, splittedData.TrainSet.Schema, TrainedModelFilePath);
        }

        private static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("SexEncoded", "Sex"), new InputOutputColumnPair("EmbarkedEncoded", "Embarked") })
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding(new[] { new InputOutputColumnPair("CabinEncoded", "Cabin") }))
                .Append(mlContext.Transforms.Text.FeaturizeText("Name_tf", "Name"))
                .Append(mlContext.Transforms.Concatenate("Features", new[] { "SexEncoded", "EmbarkedEncoded", "CabinEncoded", "Name_tf", "Pclass", "Age", "SibSp", "Parch", "Fare" }));

            var trainer = mlContext.BinaryClassification.Trainers.FastForest(numberOfLeaves: 101, minimumExampleCountPerLeaf: 1, numberOfTrees: 20, featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        private static ITransformer TrainModel(IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            WriteLine("=============== Training  model ===============");

            var model = trainingPipeline.Fit(trainingDataView);

            WriteLine("=============== End of training process ===============");
            return model;
        }

        private static void Evaluate(MLContext mlContext, IDataView testData, ITransformer trainedModel)
        {
            // Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics
            //Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            //var crossValidationResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "Survived");
            //PrintBinaryClassificationFoldsAverageMetrics(crossValidationResults);

            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions,
                labelColumnName: "Label", scoreColumnName: "Score");

            PrintMetrics(metrics);
        }

        private static void PrintBinaryClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var AccuracyValues = metricsInMultipleFolds.Select(m => m.Accuracy);
            var AccuracyAverage = AccuracyValues.Average();
            var AccuraciesStdDeviation = CalculateStandardDeviation(AccuracyValues);
            var AccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(AccuracyValues);


            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Binary Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average Accuracy:    {AccuracyAverage:0.###}  - Standard deviation: ({AccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({AccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");
        }

        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
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
