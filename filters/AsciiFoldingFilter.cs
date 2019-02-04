namespace naive_bayes
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.Text;

    public class AsciiFoldingFilter : Segment
    {
        public IEnumerable<string> Process(string input) {
            yield return RemoveDiacritics(input, true);
        }

        public static string RemoveDiacritics(string src, bool compatNorm)
        {
            ReadOnlySpan<char> input = src.Normalize(compatNorm ? NormalizationForm.FormKD : NormalizationForm.FormD).AsSpan();
            Span<char> clean = stackalloc char[input.Length];

            int p = 0;
            for (var i = 0; i < src.Length; i++) {
                char c = input[i];
                switch(CharUnicodeInfo.GetUnicodeCategory(c))
                {
                    case UnicodeCategory.NonSpacingMark:
                    case UnicodeCategory.SpacingCombiningMark:
                    case UnicodeCategory.EnclosingMark:
                        break;

                    default:
                        clean[p++] = c;
                        break;
                }
            }

            return new string(clean.Slice(0, p));
        }
    }
}
