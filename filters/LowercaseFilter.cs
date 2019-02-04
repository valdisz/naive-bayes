namespace naive_bayes
{
    using System.Collections.Generic;

    public class LowercaseFilter : Segment
    {
        public IEnumerable<string> Process(string input) {
            yield return input.ToLowerInvariant();
        }
    }
}
