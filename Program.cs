namespace naive_bayes
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using System.Text.RegularExpressions;
    using System.Threading;
    using System.Threading.Tasks;
    using Accord.MachineLearning.Bayes;
    using Accord.MachineLearning.DecisionTrees;
    using Accord.MachineLearning.DecisionTrees.Learning;
    using Accord.MachineLearning.VectorMachines.Learning;
    using Accord.Statistics.Filters;
    using Accord.Statistics.Kernels;
    using F23.StringSimilarity;
    using TinyCsvParser;
    using TinyCsvParser.Mapping;

    public class BrandRecord
    {
        public string Input { get; set; }
        public string ClassName { get; set; }
        public string SubClassName { get; set; }
    }

    public class BrandRecordMapping : CsvMapping<BrandRecord>
    {
        public BrandRecordMapping()
            : base()
        {
            MapProperty(0, x => x.Input);
            MapProperty(1, x => x.ClassName);
            MapProperty(2, x => x.SubClassName);
        }
    }

    class Vocabulory {
        public Vocabulory(IEnumerable<string> input) {
            list.AddRange(input.Distinct());
            for (var i = 0; i < list.Count; i++) {
                index.Add(list[i], i);
            }
        }

        private List<string> list = new List<string>();
        private Dictionary<string, int> index = new Dictionary<string, int>();

        public int Encode(string word) => index[word];
        public string Decode(int i) => list[i];

        public IEnumerable<int> Symbols => Enumerable.Range(0, list.Count);

        public int Count => list.Count;
    }

    class Program
    {
        static void Main(string[] args)
        {
            CsvParserOptions csvParserOptions = new CsvParserOptions(true, ';');
            BrandRecordMapping csvMapper = new BrandRecordMapping();
            CsvParser<BrandRecord> csvParser = new CsvParser<BrandRecord>(csvParserOptions, csvMapper);

            Tokenizer tokenizer = new Tokenizer(
                new SymbolTokenizer(' ', '.', ',', ';', '`', '+', '[', '?', ']', '*', '\\', ')'),
                new RegexFilter(@"\W+"),
                new LowercaseFilter(),
                new AsciiFoldingFilter(),
                // new LengthTokenFilter(len => len > 1),
                new EmptyTokenFilter()
            );

            var dataset = csvParser
                .ReadFromFile(@"training.csv", Encoding.UTF8)
                .Where(x => x.IsValid)
                .Select(x => x.Result)
                .Select(x => new { Words = tokenizer.Process(x.Input).ToArray(), x.ClassName, x.SubClassName })
                .ToArray();

            // converting string words to integer
            var words = new Vocabulory(dataset.SelectMany(x => x.Words));
            var classNames = new Vocabulory(dataset.Select(x => x.ClassName));

            // subClasses
            Dictionary<int, Vocabulory> subClassNames = new Dictionary<int, Vocabulory>();
            foreach (var c in classNames.Symbols) {
                var name = classNames.Decode(c);
                Vocabulory subClasses = new Vocabulory(dataset.Where(row => row.ClassName == name).Select(row => row.SubClassName));
                subClassNames.Add(c, subClasses);
            }

            // create inputs & outputs
            var inputs = dataset
                .Select(row => {
                    var i = new int[10];
                    Array.Fill(i, 0);

                    for (var wi = 0; wi < i.Length; wi++) {
                        if (row.Words.Length <= wi) {
                            i[wi] = 0;
                        }
                        else {
                            i[wi] = words.Encode(row.Words[wi]) + 1;
                        }
                    }

                    return i;
                })
                .ToArray();

            var outputs = dataset
                .Select(row => classNames.Encode(row.ClassName))
                .ToArray();

            var learner = new NaiveBayesLearning();
            Accord.MachineLearning.Bayes.NaiveBayes nb = learner.Learn(inputs, outputs);

            Dictionary<int, int> singleSubClass = new Dictionary<int, int>();
            Dictionary<int, Accord.MachineLearning.Bayes.NaiveBayes> subNb = new Dictionary<int, Accord.MachineLearning.Bayes.NaiveBayes>();
            foreach (var cn in classNames.Symbols) {
                var name = classNames.Decode(cn);
                var subClasses = subClassNames[cn];

                if (subClasses.Count == 1) {
                    singleSubClass.Add(cn, 0);
                    subNb.Add(cn, null);
                    continue;
                }

                var inputs2 = dataset
                    .Where(x => x.ClassName == name)
                    .Select(row => {
                        var i = new int[10];
                        Array.Fill(i, 0);

                        for (var wi = 0; wi < i.Length; wi++) {
                            if (row.Words.Length <= wi) {
                                i[wi] = 0;
                            }
                            else {
                                i[wi] = words.Encode(row.Words[wi]) + 1;
                            }
                        }

                        return i;
                    })
                    .ToArray();

                var outputs2 = dataset
                    .Where(x => x.ClassName == name)
                    .Select(row => subClasses.Encode(row.SubClassName))
                    .ToArray();

                var learner2 = new NaiveBayesLearning();
                subNb.Add(cn, learner2.Learn(inputs2, outputs2));
            }

            Stopwatch sw = Stopwatch.StartNew();
            // run
            int okBrand = 0;
            int okModel = 0;
            foreach (var row in dataset) {
                var input = row.Words.Select(w => words.Encode(w) + 1).ToArray();
                if (input.Length < 10) {
                    var len = input.Length;
                    Array.Resize(ref input, 10);
                    Array.Fill(input, 0, len, 10 - len);
                }

                var res = nb.Decide(input);
                // var probs = nb.Probabilities(input);

                var className = classNames.Decode(res);
                if (row.ClassName == className) Interlocked.Increment(ref okBrand);

                int subRes;
                if (subNb[res] != null) {
                    subRes = subNb[res].Decide(input);
                }
                else {
                    // single sub-class
                    subRes = singleSubClass[res];
                }

                var subClassName = subClassNames[res].Decode(subRes);
                if (row.SubClassName == subClassName) Interlocked.Increment(ref okModel);
            }

            sw.Stop();

            Console.WriteLine($"Brand: {okBrand / (double) dataset.Length * 100}%");
            Console.WriteLine($"Model: {okModel / (double) dataset.Length * 100}%");
            Console.WriteLine($"Speed: {sw.Elapsed / dataset.Length}/sample");
        }
    }
}
