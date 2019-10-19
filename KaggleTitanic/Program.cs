using KaggleTitanic.Model;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using static System.Console;

namespace KaggleTitanic
{
    class Program
    {

        static void Main(string[] args)
        {
            try
            {
                WriteLine("Begin");

                ReportData();

                // Best score: 0.80382
                // BinaryRegression.BinaryRegressionCommand.Execute();

                // Best score: 0.78468
                BinaryForest.BinaryForestCommand.Execute();
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

        private static void ReportData()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<TrainingPassenger>(@"data\train.csv",
                separatorChar: ',', hasHeader: true, allowQuoting: true);
            var trainingPassengers = mlContext.Data.CreateEnumerable<TrainingPassenger>(data, false).ToList();

            WriteLine($"Total passengers: {trainingPassengers.Count}");

            ReportSurvivors("Pclass", trainingPassengers.GroupBy(p => p.Pclass), trainingPassengers.Count);
            ReportSurvivors("Title", trainingPassengers.GroupBy(p => PassengerTitleMappingFactory.DetermineTitle(p.Name)), trainingPassengers.Count);
            ReportSurvivors("Sex", trainingPassengers.GroupBy(p => p.Sex), trainingPassengers.Count);
            ReportSurvivors("Age group", trainingPassengers.GroupBy(p => ((int)p.Age / 10) * 10), trainingPassengers.Count);
            ReportSurvivors("Embarked", trainingPassengers.GroupBy(p => p.Embarked), trainingPassengers.Count);
            ReportSurvivors("Siblings/Spouses", trainingPassengers.GroupBy(p => p.SibSp), trainingPassengers.Count);
            ReportSurvivors("Parents/Children", trainingPassengers.GroupBy(p => p.Parch), trainingPassengers.Count);

            WriteLine();
        }

        private static void ReportSurvivors<T>(string groupName, IEnumerable<IGrouping<T, TrainingPassenger>> passengerGroups, int totalPassengers)
        {
            foreach (var group in passengerGroups.OrderByDescending(g => (float)g.Count(p => p.Survived) / g.Count()))
            {
                var survivors = (float)group.Count(p => p.Survived);
                var survived = (survivors / totalPassengers) * 100;
                var survivalChance = (survivors / group.Count()) * 100;
                WriteLine($"{groupName} {group.Key}: contains {survived}% survivors, {survivalChance}% of this group survived");
            }
            WriteLine();
        }
    }
}
