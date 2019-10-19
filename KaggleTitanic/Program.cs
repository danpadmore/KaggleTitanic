using KaggleTitanic.BinaryRegression;
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

                BinaryRegressionCommand.Execute();
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
