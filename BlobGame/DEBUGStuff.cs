using Raylib_CsLo;
using System.Text.Json;

namespace BlobGame;
internal static class DEBUGStuff {

    public static void DEBUG_SerializeText() {
        Dictionary<string, string> scrolls = new Dictionary<string, string> {
            { "Classic", "Play the classic mode!\nCombine pieces and gain points." },
        };
        string json = JsonSerializer.Serialize(scrolls, new JsonSerializerOptions() { WriteIndented = true });
        File.WriteAllText(@"G:\Coding\C#\BlobGame\BlobGame\Resources\References\Texts\game_mode_descriptions.json", json);

        Dictionary<string, string> d2 = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(@"G:\Coding\C#\BlobGame\BlobGame\Resources\References\Texts\game_mode_descriptions.json"));
    }

    public static void DEBUG_PrintNPatchJson() {
        NPatchInfo npi = new NPatchInfo(new Rectangle(0, 0, 500, 400), 65, 50, 856, 167, NPatchLayout.NPATCH_NINE_PATCH);

        string json = JsonSerializer.Serialize<Dictionary<string, int>>(new() {
            { "left", npi.left },
            { "right", npi.right },
            { "top", npi.top },
            { "bottom", npi.bottom },
            { "layout", npi.layout },
        }, new JsonSerializerOptions() { WriteIndented = true });

        Dictionary<string, int>? dict = JsonSerializer.Deserialize<Dictionary<string, int>>(json);
        NPatchInfo npi2 = new NPatchInfo(new Rectangle(0, 0, 500, 400), dict["left"], dict["right"], dict["top"], dict["bottom"], (NPatchLayout)dict["layout"]);
    }
}
