using BlobGame.App;
using BlobGame.Game.Scenes;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Globalization;
using System.Numerics;

namespace BlobGame.Game.Tutorial;
internal sealed class TutorialDisplay {
    private const float HOLD_TIME = 0.5f;

    internal GameScene Scene { get; }
    internal string GameModeKey { get; }
    internal string AvatarName { get; set; }

    private TextResource TutorialTextResource { get; set; }

    public IReadOnlyList<TutorialStage> Stages { get; set; }
    private int CurrentStageIndex { get; set; }
    private TutorialStage? CurrentStage { get; set; }

    public bool IsFinished => CurrentStageIndex >= Stages?.Count;

    private float HoldTime { get; set; }
    private bool AdvanceStage { get; set; }
    public float AdvanceProgress => HoldTime / HOLD_TIME;

    public TutorialDisplay(GameScene scene, string gameModeKey) {
        Scene = scene;
        GameModeKey = gameModeKey;

        CurrentStageIndex = -1;
        HoldTime = 0;
        AdvanceStage = false;
    }

    internal void Load() {
        TutorialTextResource = ResourceManager.TextLoader.Get($"{GameModeKey}_tutorial");
        TutorialTextResource.WaitForLoad();
        int stageCount = int.Parse(TutorialTextResource.Resource["stages"], CultureInfo.InvariantCulture);
        AvatarName = TutorialTextResource.Resource["avatar_name"];

        ResourceManager.TextureLoader.Load("avatar_idle");
        ResourceManager.TextureLoader.Load("avatar_blink_0");
        ResourceManager.TextureLoader.Load("avatar_blink_1");
        ResourceManager.TextureLoader.Load("avatar_talk_0");
        ResourceManager.TextureLoader.Load("avatar_talk_1");
        ResourceManager.TextureLoader.Load("avatar_talk_2");
        ResourceManager.TextureLoader.Load("lmb");
        ResourceManager.TextureLoader.Load("pointer");
        ResourceManager.TextureLoader.Load("speechbubble");

        for (int i = 0; i < stageCount; i++) {
            string key = $"{GameModeKey}_tutorial_{i}";
            if (ResourceManager.SoundLoader.ResourceExists(key))
                ResourceManager.SoundLoader.Load(key);
        }

        ResourceManager.WaitForLoading();

        TutorialStage[] stages = new TutorialStage[stageCount];
        for (int i = 0; i < stageCount; i++)
            stages[i] = CreateStage(i);
        Stages = stages;

        LoadNextStage();
    }

    internal void Draw(float dT) {
        if (IsFinished)
            return;

        Raylib.DrawRectangleRec(new Rectangle(0, 0, Application.BASE_WIDTH, Application.BASE_HEIGHT), Raylib.WHITE.ChangeAlpha(64));

        if (!CurrentStage!.IsFadeInFinished)
            CurrentStage.DrawFadeIn();
        else if (!AdvanceStage)
            CurrentStage.Draw(dT);
        else if (!CurrentStage.IsFadeOutFinished)
            CurrentStage.DrawFadeOut();
    }

    internal void Update(float dT) {
        if (IsFinished)
            return;

        if (CurrentStage != null &&
            CurrentStage.IsFadeInFinished &&
            !AdvanceStage &&
            !Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT]) {

            if (Input.IsMouseButtonDown(MouseButton.MOUSE_BUTTON_LEFT))
                HoldTime += dT;
            else
                HoldTime = 0;
        }

        if (!AdvanceStage && HoldTime >= HOLD_TIME) {
            Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT] = true;

            HoldTime = 0;
            AdvanceStage = true;
        }

        if (CurrentStage != null && CurrentStage.IsFadeOutFinished && AdvanceStage) {
            LoadNextStage();
            AdvanceStage = false;
        }
    }

    internal void Unload() {
        CurrentStage?.Unload();
    }

    private void LoadNextStage() {
        CurrentStage?.Unload();
        CurrentStageIndex++;

        if (CurrentStageIndex >= Stages.Count)
            return;

        CurrentStage = Stages[CurrentStageIndex];

        CurrentStage?.Load();
    }

    private TutorialStage CreateStage(int stageIndex) {
        string text = ParseStageText(TutorialTextResource.Resource[$"{stageIndex}_text"]);
        float[] pointerPos = TutorialTextResource.Resource[$"{stageIndex}_pointerPos"].Split(",", StringSplitOptions.TrimEntries).Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        float pointerRot = float.Parse(TutorialTextResource.Resource[$"{stageIndex}_pointerRot"], CultureInfo.InvariantCulture);
        float avatarX = float.Parse(TutorialTextResource.Resource[$"{stageIndex}_avatarX"], CultureInfo.InvariantCulture);
        float[] speechBubblePos = TutorialTextResource.Resource[$"{stageIndex}_speechbubblePos"].Split(",", StringSplitOptions.TrimEntries).Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        int speechFrames = int.Parse(TutorialTextResource.Resource[$"{stageIndex}_speechFrames"], CultureInfo.InvariantCulture);
        int overlayIndex = int.Parse(TutorialTextResource.Resource[$"{stageIndex}_overlayIndex"], CultureInfo.InvariantCulture);

        return new TutorialStage(
            this, stageIndex,
            text,
            new Vector2(pointerPos[0], pointerPos[1]), pointerRot,
            avatarX, new Vector2(speechBubblePos[0], speechBubblePos[1]),
            speechFrames, overlayIndex);
    }

    private string ParseStageText(string text) {

        int index = -2;
        while (index + 2 < text.Length && (index = text.IndexOf("$$", index + 2)) != -1) {
            int endIndex = text.IndexOf("$$", index + 2);

            if (endIndex == -1)
                continue;

            string key = text.Substring(index + 2, endIndex - index - 2);
            string[] components = key.Split("/", StringSplitOptions.TrimEntries);

            if (components.Length != 2)
                continue;

            if (components[0] != "hk")
                continue;

            string hotkey = components[1];
            string hotkeyText = Input.GetHotkeyDisplayString(hotkey);

            if (string.IsNullOrWhiteSpace(hotkeyText))
                continue;

            text = text.Replace($"$${key}$$", hotkeyText);

            index = endIndex + 2;
        }

        return text;
    }
}

