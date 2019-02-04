namespace naive_bayes
{
    using System.Collections.Generic;
    using System.Linq;

    public class Tokenizer : Segment {
        public Tokenizer(params Segment[] segments) {
            this.segments = segments.ToList();
        }

        private readonly List<Segment> segments;

        public IEnumerable<string> Process(string input) {
            IEnumerable<string> result = new[] { input };
            
            foreach (var t in segments) {
                result = result.SelectMany(s => t.Process(s));
            }

            return result;
        }
    }
}
