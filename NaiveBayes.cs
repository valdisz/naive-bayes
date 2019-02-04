namespace naive_bayes
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public class NaiveBayes {
        public NaiveBayes(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        private readonly Tokenizer tokenizer;

        public int TotalRecordCount;

        public readonly List<string> ClassList = new List<string>();
        public readonly List<int> ClassCount = new List<int>();
        public readonly List<Dictionary<string, int>> ClassTokenCount = new List<Dictionary<string, int>>();
        public double[] ClassPropb;

        public readonly HashSet<string> Vocab = new HashSet<string>();

        public void Fit(IEnumerable<(string text, string className)> records) {
            foreach (var rec in records) {
                TotalRecordCount++;

                var tokens = tokenizer.Process(rec.text).ToArray();;
                Vocab.UnionWith(tokens);

                var classIndex = ClassList.IndexOf(rec.className);
                if (classIndex < 0) {
                    ClassList.Add(rec.className);
                    ClassCount.Add(0);
                    ClassTokenCount.Add(new Dictionary<string, int>());

                    classIndex = ClassList.Count - 1;
                }

                ClassCount[classIndex]++;
                
                var tc = ClassTokenCount[classIndex];
                foreach (var t in tokens) {
                    if (!tc.ContainsKey(t)) tc[t] = 0;

                    tc[t]++;
                }
            }

            ClassPropb = new double[ClassList.Count];
            for (var i = 0; i < ClassPropb.Length; i++) {
                ClassPropb[i] = Math.Log(ClassCount[i] / (double) TotalRecordCount);
            }
        }

        public (string, double)[] Predict(string input) {
            (int inClass, int outClass) getTokenCount(int classIndex, string token) {
                var inClass = 0;
                var outClass = 0;

                for (var i = 0; i < ClassList.Count; i++) {
                    if (i == classIndex) {
                        inClass += ClassTokenCount[i].TryGetValue(token, out var cnt)
                            ? cnt
                            : 0;
                    }
                    else {
                        outClass += ClassTokenCount[i].TryGetValue(token, out var cnt)
                            ? cnt
                            : 0;
                    }
                }

                return (inClass, outClass);
            }

            var tokens = tokenizer.Process(input).ToArray();
            double[] scoreYes = new double[ClassList.Count];
            double[] scoreNo = new double[ClassList.Count];
            Array.Fill(scoreYes, 0.0);
            Array.Fill(scoreNo, 0.0);

            foreach (var t in tokens) {
                if (!Vocab.Contains(t)) continue;

                for (var i = 0; i < ClassList.Count; i++) {
                    var (yesClassTokenCount, noClassTokenCount) = getTokenCount(i, t);
                    var yesClassTotalCount = ClassCount[i];
                    var noClassTotalCount = TotalRecordCount - yesClassTotalCount;

                    var yesP = Math.Log((yesClassTokenCount + 1) / (double) (yesClassTotalCount + Vocab.Count));
                    var noP = Math.Log((noClassTokenCount + 1) / (double) (noClassTotalCount + Vocab.Count));

                    scoreYes[i] += yesP;
                    scoreNo[i] += noP;
                }
            }

            // adjust result with class porbabilities
            // for (var i = 0; i < ClassList.Count; i++) {
            //     scoreYes[i] += ClassPropb[i];
            //     scoreNo[i] += ClassPropb.Where((_, j) => i != j).Sum();
            // }

            return scoreYes.Zip(scoreNo, (yes, no) => {
                    var total = Math.Abs(yes) + Math.Abs(no);
                    return (yes + total) / (yes + no + total * 2);
                })
                .Select((p, i) => (ClassList[i], p))
                .OrderByDescending(x => x.Item2)
                .ToArray();
        }
    }
}
