namespace naive_bayes
{
    using System.Collections.Generic;
    using System.Linq;

    public class SymbolTokenizer : Segment {
        public SymbolTokenizer(params char[] symbols) {
            this.symbols = symbols;
        }

        private readonly char[] symbols;

        public IEnumerable<string> Process(string input) {
            return input.Split(symbols).Where(x => !string.IsNullOrWhiteSpace(x));
        }
    }
}
