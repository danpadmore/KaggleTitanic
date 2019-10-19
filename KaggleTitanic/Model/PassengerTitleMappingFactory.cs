using Microsoft.ML.Transforms;
using System;

namespace KaggleTitanic.Model
{
    [CustomMappingFactoryAttribute(ContractName)]
    public class PassengerTitleMappingFactory : CustomMappingFactory<PassengerName, PassengerTitle>
    {
        public const string ContractName = "PassengerTitle";

        private static void CustomAction(PassengerName input, PassengerTitle output)
        {
            output.LastName = input.Name.Substring(0, input.Name.IndexOf(','));
            output.Title = DetermineTitle(input.Name);
        }

        public override Action<PassengerName, PassengerTitle> GetMapping()
        {
            return CustomAction;
        }

        public static string DetermineTitle(string name)
        {
            if (name.Contains("Mr.")) return "Mr.";
            if (name.Contains("Mrs.")) return "Mrs.";
            if (name.Contains("Miss.")) return "Miss.";
            if (name.Contains("Master.")) return "Master.";

            return null;
        }
    }

    public class PassengerTitle
    {
        public string Title { get; set; }
        public string LastName { get; set; }
    }
}
