using Microsoft.ML.Data;

namespace KaggleTitanic
{
    public class Passenger
    {
        [LoadColumn(0)]
        public short PassengerId { get; set; }

        /// <summary>
        /// Passenger's class (1st, 2nd, 3rd)
        /// </summary>
        [LoadColumn(1)]
        public float Pclass { get; set; }

        [LoadColumn(2)]
        public string Name { get; set; }

        /// <summary>
        /// male, female
        /// </summary>
        [LoadColumn(3)]
        public string Sex { get; set; }

        [LoadColumn(4)]
        public float Age { get; set; }

        /// <summary>
        /// Number of siblings/spouses aboard the Titanic
        /// </summary>
        [LoadColumn(5)]
        public float SibSp { get; set; }

        /// <summary>
        /// Number of parents/children aboard the Titanic
        /// </summary>
        [LoadColumn(6)]
        public float Parch { get; set; }

        [LoadColumn(7)]
        public string Ticket { get; set; }

        [LoadColumn(8)]
        public float Fare { get; set; }

        [LoadColumn(9)]
        public string Cabin { get; set; }

        /// <summary>
        /// C = Cherbourg, S = Southampton, Q = Queenstown
        /// </summary>
        [LoadColumn(10)]
        public string Embarked { get; set; }
    }
}
