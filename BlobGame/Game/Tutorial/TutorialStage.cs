using BlobGame.Audio;
using BlobGame.Drawing;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Tutorial;
internal sealed class TutorialStage {
    //const int AVATAR_WIDTH = 944;
    //const int AVATAR_HEIGHT = 1432;
    const int AVATAR_WIDTH = 768;
    const int AVATAR_HEIGHT = 768;
    const float FONT_SIZE = 50;

    private TutorialDisplay Tutorial { get; }
    private int StageIndex { get; }
    public string Text { get; }
    public Vector2 PointerPos { get; }
    public float PointerRot { get; }
    public float AvatarX { get; }
    public Vector2 SpeechBubblePos { get; }
    public float HintX { get; }

    internal bool IsFadeInFinished => !HasFadeIn || AnimatedAvatarFadeIn!.IsFinished;
    internal bool IsFadeOutFinished => !HasFadeOut || AnimatedAvatarFadeOut!.IsFinished;

    private TextureResource LMBTexture { get; set; }
    private TextureResource PointerTexture { get; set; }
    private NPatchTextureResource SpeechbubbleTexture { get; set; }
    private SoundResource SpeechResource { get; set; }

    private TextureResource AvatarIdleTexture { get; set; }
    private TextureResource AvatarBlink0Texture { get; set; }
    private TextureResource AvatarBlink1Texture { get; set; }
    private TextureResource AvatarTalk0Texture { get; set; }
    private TextureResource AvatarTalk1Texture { get; set; }
    private TextureResource AvatarTalk2Texture { get; set; }

    private AnimatedTexture? AnimatedAvatarFadeIn { get; set; }
    private AnimatedTexture? AnimatedAvatarFadeOut { get; set; }
    private AnimatedTexture AnimatedPointer { get; set; }
    private AnimatedTexture AnimatedSpeechbubble { get; set; }
    private FrameAnimator AvatarAnimator { get; set; }

    private bool HasFadeIn => StageIndex == 0 || AvatarX != Tutorial.Stages[StageIndex - 1].AvatarX;
    private bool HasFadeOut => StageIndex == Tutorial.Stages.Count - 1 || AvatarX != Tutorial.Stages[StageIndex + 1].AvatarX;

    private Vector2 PointerAnimationDirection { get; }
    private bool PlayedSound { get; set; }
    private int TalkFrames { get; set; }

    public TutorialStage(TutorialDisplay tutorial, int stageIndex, string text, Vector2 pointerPos, float pointerRot, float avatarX, Vector2 speechBubblePos, float hintX) {
        Tutorial = tutorial;
        StageIndex = stageIndex;
        Text = text;
        PointerPos = pointerPos;
        PointerRot = pointerRot;
        AvatarX = avatarX;
        SpeechBubblePos = speechBubblePos;
        HintX = hintX;
        PointerAnimationDirection = new Vector2(MathF.Cos(MathF.PI / 2f + PointerRot * RayMath.DEG2RAD), MathF.Sin(MathF.PI / 2 + PointerRot * RayMath.DEG2RAD));

        TalkFrames = 5;
        AvatarAnimator = new FrameAnimator(2f / 24f);
    }

    internal void Load() {
        LMBTexture = ResourceManager.TextureLoader.Get("lmb");
        PointerTexture = ResourceManager.TextureLoader.Get("pointer");
        SpeechbubbleTexture = ResourceManager.NPatchTextureLoader.Get("speechbubble");
        SpeechResource = ResourceManager.SoundLoader.Get($"{Tutorial.GameModeKey}_tutorial_{StageIndex}");

        AvatarIdleTexture = ResourceManager.TextureLoader.Get("avatar_idle");
        AvatarBlink0Texture = ResourceManager.TextureLoader.Get("avatar_blink_0");
        AvatarBlink1Texture = ResourceManager.TextureLoader.Get("avatar_blink_1");
        AvatarTalk0Texture = ResourceManager.TextureLoader.Get("avatar_talk_0");
        AvatarTalk1Texture = ResourceManager.TextureLoader.Get("avatar_talk_1");
        AvatarTalk2Texture = ResourceManager.TextureLoader.Get("avatar_talk_2");

        AvatarAnimator.AddFrameKey("idle", AvatarIdleTexture);
        AvatarAnimator.AddFrameKey("blink0", AvatarBlink0Texture);
        AvatarAnimator.AddFrameKey("blink1", AvatarBlink1Texture);
        AvatarAnimator.AddFrameKey("talk0", AvatarTalk0Texture);
        AvatarAnimator.AddFrameKey("talk1", AvatarTalk1Texture);
        AvatarAnimator.AddFrameKey("talk2", AvatarTalk2Texture);
        AvatarAnimator.AddFrameSequence("idle", 3, "idle", "idle", "idle", "idle", "idle");
        AvatarAnimator.AddFrameSequence("idle", 1, "idle", "blink0", "blink1", "blink0", "idle");
        AvatarAnimator.AddFrameSequence("talk", 1, "talk2", "talk1", "talk0", "talk1", "talk2");
        AvatarAnimator.SetDefaultSequence("idle");

        PlayedSound = false;

        AnimatedPointer = new AnimatedTexture(
            PointerTexture,
            0.5f,
            PointerPos,
            new Vector2(),
            Vector2.One / 2f,
            PointerRot * RayMath.DEG2RAD) {
            PositionAnimator = t => PointerAnimationDirection * 10 * -MathF.Sin(MathF.Tau * t)
        };

        Vector2 textSize = Raylib.MeasureTextEx(Renderer.GuiFont.Resource, Text, FONT_SIZE, FONT_SIZE / 16f);

        AnimatedSpeechbubble = new AnimatedTexture(
            SpeechbubbleTexture,
            2,
            SpeechBubblePos,
            textSize * 1.5f,
            new Vector2(0.5f, 0.5f)) {
            ScaleAnimator = t => Vector2.One + new Vector2(0.05f, 0.05f) * GetSpeechbubbleScaleT(t),
            RotationAnimator = t => MathF.PI / 128 * GetSpeechbubbleRotationT(t)
        };

        if (HasFadeIn) {
            AnimatedAvatarFadeIn = new AnimatedTexture(
                AvatarIdleTexture,
                0.65f,
                new Vector2(AvatarX, Application.BASE_HEIGHT + 1432),
                new Vector2(AVATAR_WIDTH, AVATAR_HEIGHT),
                new Vector2(0, 1)) {
                PositionAnimator = t => new Vector2(0, -GetAvatarPositionT(t) * 2 * AVATAR_HEIGHT)
            };
        }

        if (HasFadeOut) {
            AnimatedAvatarFadeOut = new AnimatedTexture(
                AvatarIdleTexture,
                0.65f,
                new Vector2(AvatarX, Application.BASE_HEIGHT + AVATAR_HEIGHT / 2),
                new Vector2(AVATAR_WIDTH, AVATAR_HEIGHT),
                new Vector2(0, 1)) {
                PositionAnimator = t => new Vector2(0, GetAvatarPositionT(t) * 2 * AVATAR_HEIGHT)
            };
        }
    }

    internal void Unload() {
        SpeechResource.Unload();
    }

    internal void DrawFadeIn() {
        if (!HasFadeIn)
            return;

        if (AnimatedAvatarFadeIn!.IsReady)
            AnimatedAvatarFadeIn.Start();
        AnimatedAvatarFadeIn.Draw();
    }

    internal void DrawFadeOut() {
        if (!HasFadeOut)
            return;

        if (AnimatedAvatarFadeOut!.IsReady)
            AnimatedAvatarFadeOut.Start();
        AnimatedAvatarFadeOut.Draw();
    }

    internal void Draw(float dT) {
        if (!PlayedSound) {
            AudioManager.PlaySound($"{Tutorial.GameModeKey}_tutorial_{StageIndex}");
            PlayedSound = true;

            AvatarAnimator.StartSequence("talk");
        }

        if (TalkFrames > 0 && AvatarAnimator.IsReady) {
            AvatarAnimator.StartSequence("talk");
            TalkFrames -= 1;
        }

        AvatarAnimator.Draw(dT, new Rectangle(AvatarX, Application.BASE_HEIGHT - AVATAR_HEIGHT, AVATAR_WIDTH, AVATAR_HEIGHT), 0, new Vector2(0, 0), Raylib.WHITE);

        if (AnimatedPointer.IsReady)
            AnimatedPointer.Start();

        AnimatedPointer.Draw();

        if (AnimatedPointer.IsFinished)
            AnimatedPointer.Start();

        DrawSpeechBubble();

        Renderer.GuiFont.Draw(
            Text,
            FONT_SIZE,
            ResourceManager.ColorLoader.Get("font_dark"),
            SpeechBubblePos,
            new Vector2(0.5f, 0.5f));

        DrawLMBHint(HintX);
    }

    private void DrawSpeechBubble() {
        if (AnimatedSpeechbubble.IsReady)
            AnimatedSpeechbubble.Start();

        AnimatedSpeechbubble.Draw();

        if (AnimatedSpeechbubble.IsFinished)
            AnimatedSpeechbubble.Start();
    }

    protected void DrawLMBHint(float x) {
        LMBTexture.Draw(
            new Vector2(x, Application.BASE_HEIGHT - 50),
            new Vector2(0, 1),
            new Vector2(0.4f, 0.4f));

        Renderer.GuiFont.Draw(
            "Hold LMB to continue!",
            80,
            ResourceManager.ColorLoader.Get("font_dark"),
            new Vector2(x + 125, Application.BASE_HEIGHT - 150));
    }

    private float GetAvatarPositionT(float t) {
        float tmp = 1.3f * t - 1;
        return 1.1f * (-(tmp * tmp) + 1);
    }

    private float GetSpeechbubbleScaleT(float t) {
        return 0.5f * (MathF.Sin(MathF.Tau * t - MathF.PI / 2) + 1);
    }

    private float GetSpeechbubbleRotationT(float t) {
        return -MathF.Sin(MathF.Tau * t);
    }
}