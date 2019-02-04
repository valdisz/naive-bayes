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
            bag.UnionWith(input);
            list.AddRange(bag);
            for (var i = 0; i < list.Count; i++) {
                index.Add(list[i], i);
            }
        }

        private List<string> list = new List<string>();
        private Dictionary<string, int> index = new Dictionary<string, int>();
        private HashSet<string> bag = new HashSet<string>();

        public int Encode(string word) => index[word];
        public string Decode(int i) => list[i];
        public bool Has(string s) => bag.Contains(s);

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

            var tempDataset = csvParser
                .ReadFromFile(@"training.csv", Encoding.UTF8)
                .Where(x => x.IsValid)
                .Select(x => x.Result)
                .Select(x => (
                    words: tokenizer.Process(x.Input).ToArray(),
                    className: x.ClassName.ToLowerInvariant(),
                    subClassName: x.SubClassName.ToLowerInvariant()
                ))
                .ToArray();

            var dataset = tempDataset
                .Concat(tempDataset.Select(row => (
                    words: tokenizer.Process($"{row.className} {row.subClassName}").ToArray(),
                    className: row.className,
                    subClassName: row.subClassName
                )))
                .ToArray();

            // converting string words to integer
            var wordsVocab = new Vocabulory(dataset.SelectMany(x => x.words));
            var classNames = new Vocabulory(dataset.Select(x => x.className));
            var subClassNames = new Vocabulory(dataset.Select(x => x.subClassName));

            // subClasses
            Dictionary<int, Vocabulory> scopedSubClassNames = new Dictionary<int, Vocabulory>();
            foreach (var c in classNames.Symbols) {
                var name = classNames.Decode(c);
                Vocabulory subClasses = new Vocabulory(dataset.Where(row => row.className == name).Select(row => row.subClassName));
                scopedSubClassNames.Add(c, subClasses);
            }

            const int WORDS_SZ = 10;
            const int SINGLE_CLASS_SZ = 2;
            const int CLASS_SZ = SINGLE_CLASS_SZ * 2;
            const int INPUT_SZ = CLASS_SZ + WORDS_SZ;

            int[] exactMatch(Vocabulory v, string[] list) {
                int[] arr = new int[SINGLE_CLASS_SZ];
                Array.Fill(arr, 0);

                var found = list.Where(x => v.Has(x)).Select(x => v.Encode(x)).Take(arr.Length).ToArray();
                for (var i = 0; i < found.Length; i++) {
                    arr[i] = found[i];
                }

                return arr;
            }

            int[] makeInputVector((string[] words, string className, string subClassName) row) {
                var inArray = new int[INPUT_SZ];
                Array.Fill(inArray, 0);

                Array.Copy(exactMatch(classNames, row.words), 0, inArray, 0, SINGLE_CLASS_SZ);
                Array.Copy(exactMatch(subClassNames, row.words), 0, inArray, SINGLE_CLASS_SZ, SINGLE_CLASS_SZ);

                var wordArray = row.words.Select(x => wordsVocab.Encode(x) + 1).ToArray();
                var len = Math.Min(wordArray.Length, WORDS_SZ);
                for (var i = 0; i < len; i++) {
                    inArray[i + CLASS_SZ] = wordArray[i];
                }

                return inArray;
            }

            // create inputs & outputs
            var inputs = dataset
                .Select(row => makeInputVector(row))
                .ToArray();

            var outputs = dataset
                .Select(row => classNames.Encode(row.className))
                .ToArray();

            var learner = new NaiveBayesLearning();
            Accord.MachineLearning.Bayes.NaiveBayes nb = learner.Learn(inputs, outputs);

            Dictionary<int, int> singleSubClass = new Dictionary<int, int>();
            Dictionary<int, Accord.MachineLearning.Bayes.NaiveBayes> subNb = new Dictionary<int, Accord.MachineLearning.Bayes.NaiveBayes>();
            foreach (var cn in classNames.Symbols) {
                var name = classNames.Decode(cn);
                var subClasses = scopedSubClassNames[cn];

                if (subClasses.Count == 1) {
                    singleSubClass.Add(cn, 0);
                    subNb.Add(cn, null);
                    continue;
                }

                var inputs2 = dataset
                    .Where(x => x.className == name)
                    .Select(row => makeInputVector(row))
                    .ToArray();

                var outputs2 = dataset
                    .Where(x => x.className == name)
                    .Select(row => subClasses.Encode(row.subClassName))
                    .ToArray();

                var learner2 = new NaiveBayesLearning();
                subNb.Add(cn, learner2.Learn(inputs2, outputs2));
            }

            Stopwatch sw = Stopwatch.StartNew();
            // run
            int okBrand = 0;
            int okModel = 0;
            foreach (var row in dataset) {
                var wordInput = row.words.Select(w => wordsVocab.Encode(w) + 1).ToArray();
                var classInput = exactMatch(classNames, row.words);
                var subClassInput = exactMatch(subClassNames, row.words);

                int[] input = new int[INPUT_SZ];
                Array.Fill(input, 0);
                Array.Copy(classInput, 0, input, 0, SINGLE_CLASS_SZ);
                Array.Copy(subClassInput, 0, input, SINGLE_CLASS_SZ, SINGLE_CLASS_SZ);
                Array.Copy(wordInput, 0, input, CLASS_SZ, Math.Min(wordInput.Length, WORDS_SZ));

                var res = nb.Decide(input);
                // var probs = nb.Probabilities(input);

                var className = classNames.Decode(res);
                if (row.className == className) Interlocked.Increment(ref okBrand);

                int subRes;
                if (subNb[res] != null) {
                    subRes = subNb[res].Decide(input);
                }
                else {
                    // single sub-class
                    subRes = singleSubClass[res];
                }

                var subClassName = scopedSubClassNames[res].Decode(subRes);
                if (row.subClassName == subClassName) Interlocked.Increment(ref okModel);

                // if (row.className != className || row.subClassName != subClassName) {
                //     Console.WriteLine($"{string.Join(" ", row.words)}\tgot {className} {subClassName}\t must be {row.className} {row.subClassName}");
                // }
            }

            sw.Stop();

            Console.WriteLine($"Brand: {okBrand / (double) dataset.Length * 100}%");
            Console.WriteLine($"Model: {okModel / (double) dataset.Length * 100}%");
            Console.WriteLine($"Speed: {sw.Elapsed / dataset.Length}/sample");
        }
    }
}
