namespace naive_bayes
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Text.RegularExpressions;
    using System.Threading;
    using System.Threading.Tasks;
    using Accord.MachineLearning.Bayes;
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
                .ToArray();

            NaiveBayes topPredictor = new NaiveBayes(tokenizer);
            topPredictor.Fit(dataset.Select(row => (row.Input, row.ClassName)));

            Dictionary<string, NaiveBayes> subPredictor = new Dictionary<string, NaiveBayes>();
            foreach (var className in topPredictor.ClassList) {
                var subDataset = dataset
                    .Where(row => row.ClassName == className)
                    .Select(row => (row.Input, row.SubClassName));

                var predictor = new NaiveBayes(tokenizer);
                predictor.Fit(subDataset);

                subPredictor[className] = predictor;
            }

            int okBrand = 0;
            int okModel = 0;
            Parallel.ForEach(dataset, row => {
                var classNames = topPredictor.Predict(row.Input);
                var topClassName = classNames.First().Item1;

                var subClassNames = subPredictor[topClassName].Predict(row.Input);
                var topSubClassName = subClassNames.First().Item1;

                if (topClassName == row.ClassName) Interlocked.Increment(ref okBrand);
                if (topSubClassName == row.SubClassName) Interlocked.Increment(ref okModel);
            });

            Console.WriteLine($"Brand: {okBrand / (double) dataset.Length * 100}%");
            Console.WriteLine($"Model: {okModel / (double) dataset.Length * 100}%");
        }
    }
}
