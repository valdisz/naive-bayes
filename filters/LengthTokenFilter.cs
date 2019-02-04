namespace naive_bayes
{
    using System;
    using System.Collections.Generic;

    public class LengthTokenFilter : Segment {
        public LengthTokenFilter(Func<int, bool> predicate) {
            this.predicate = predicate;
        }

        private readonly Func<int, bool> predicate;

        public IEnumerable<string> Process(string input) {
            if (predicate(input.Length)) yield return input;
        }
    }
}
