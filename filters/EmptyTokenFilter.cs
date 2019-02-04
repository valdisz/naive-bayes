namespace naive_bayes
{
    using System.Collections.Generic;

    public class EmptyTokenFilter : Segment {
        public EmptyTokenFilter() {
        }

        public IEnumerable<string> Process(string input) {
            if (!string.IsNullOrWhiteSpace(input)) yield return input;
        }
    }
}
