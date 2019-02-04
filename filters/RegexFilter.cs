namespace naive_bayes
{
    using System.Collections.Generic;
    using System.Text.RegularExpressions;

    public class RegexFilter : Segment {
        public RegexFilter(string pattern) {
            this.pattern = new Regex(pattern);
        }

        private readonly Regex pattern;

        public IEnumerable<string> Process(string input)
        {
            var s = pattern.Replace(input, "");
            if (!string.IsNullOrWhiteSpace(s)) yield return s;
        }
    }
}
