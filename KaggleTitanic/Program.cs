using System;
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

    }
}
