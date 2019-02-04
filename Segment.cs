namespace naive_bayes
{
    using System.Collections.Generic;

    public interface Segment {
        IEnumerable<string> Process(string input);
    }
}
