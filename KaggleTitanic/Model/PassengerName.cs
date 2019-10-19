namespace KaggleTitanic.Model
{
    /// <summary>
    /// Workaround to share same type for training and predicting, which is required for custom mapping to title
    /// </summary>
    public class PassengerName
    {
        public virtual string Name { get; set; }
    }
}
