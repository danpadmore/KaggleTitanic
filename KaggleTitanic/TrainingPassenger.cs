using Microsoft.ML.Data;

namespace KaggleTitanic
{
    public class TrainingPassenger
    {
        [LoadColumn(0)]
        public float PassengerId { get; set; }

        /// <summary>
        /// Survived (1) or died (0)
        /// </summary>
        [LoadColumn(1), ColumnName("Label")]
        public bool Survived { get; set; }

        /// <summary>
        /// Passenger's class (1st, 2nd, 3rd)
        /// </summary>
        [LoadColumn(2)]
        [Feature]
        public float Pclass { get; set; }

        [LoadColumn(3)]
        [Feature]
        public string Name { get; set; }

        /// <summary>
        /// male, female
        /// </summary>
        [LoadColumn(4)]
        [Feature]
        public string Sex { get; set; }

        [LoadColumn(5)]
        [Feature]
        public float Age { get; set; }

        /// <summary>
        /// Number of siblings/spouses aboard the Titanic
        /// </summary>
        [LoadColumn(6)]
        [Feature]
        public float SibSp { get; set; }

        /// <summary>
        /// Number of parents/children aboard the Titanic
        /// </summary>
        [LoadColumn(7)]
        [Feature]
        public float Parch { get; set; }

        [LoadColumn(8)]
        public string Ticket { get; set; }

        [LoadColumn(9)]
        public float Fare { get; set; }

        [LoadColumn(10)]
        [Feature]
        public string Cabin { get; set; }

        /// <summary>
        /// C = Cherbourg, S = Southampton, Q = Queenstown
        /// </summary>
        [LoadColumn(11)]
        [Feature]
        public string Embarked { get; set; }
    }
}
