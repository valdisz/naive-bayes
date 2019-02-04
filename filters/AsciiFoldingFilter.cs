namespace naive_bayes
{
    using System.Collections.Generic;
    using System.Globalization;
    using System.Text;

    public class AsciiFoldingFilter : Segment
    {
        public IEnumerable<string> Process(string input) {
            yield return RemoveDiacritics(input, true);
        }

        public static IEnumerable<char> RemoveDiacriticsEnum(string src, bool compatNorm)
        {
            foreach(char c in src.Normalize(compatNorm ? NormalizationForm.FormKD : NormalizationForm.FormD))
            switch(CharUnicodeInfo.GetUnicodeCategory(c))
            {
                case UnicodeCategory.NonSpacingMark:
                case UnicodeCategory.SpacingCombiningMark:
                case UnicodeCategory.EnclosingMark:
                    break;

                default:
                    yield return c;
                    break;
            }
        }

        public static string RemoveDiacritics(string src, bool compatNorm)
        {
            StringBuilder sb = new StringBuilder();
            foreach(char c in RemoveDiacriticsEnum(src, compatNorm))
                sb.Append(c);
            return sb.ToString();
        }
    }
}
