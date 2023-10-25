namespace BlobGame.Game.Util;

public interface IReadOnlyGameObjectsCollection {
    int Count { get; }
    bool IsReadOnly { get; }

    GameObject? FindByName(string name);
    bool Contains(GameObject item);
    void Enumerate(Action<GameObject> iterationAction);
}

internal class GameObjectCollection : IReadOnlyGameObjectsCollection {

    public int Count => List.Count;
    public bool IsReadOnly => false;

    private List<GameObject> List { get; }
    private object Lock { get; }
    private bool IsDirty { get; set; }

    public GameObjectCollection() {
        List = new List<GameObject>();
        Lock = new object();
        IsDirty = false;
    }

    public GameObject? FindByName(string name) {
        lock (Lock) {
            return List.SingleOrDefault(item => item.Name == name);
        }
    }

    public void Enumerate(Action<GameObject> iterationAction) {
        lock (Lock) {
            if (IsDirty) {
                List.Sort((a, b) => a.ZIndex.CompareTo(b.ZIndex));
                IsDirty = false;
            }

            foreach (GameObject item in List) {
                iterationAction(item);
            }
        }
    }

    public void Add(GameObject item) {
        lock (Lock) {
            List.Add(item);
            IsDirty = true;
        }
    }

    public void AddRange(IEnumerable<GameObject> en) {
        lock (Lock) {
            List.AddRange(en);
            IsDirty = true;
        }
    }

    public bool Remove(GameObject item) {
        lock (Lock) {
            return List.Remove(item);
        }
    }

    public void Clear() {
        lock (Lock) {
            List.Clear();
        }
    }

    public bool Contains(GameObject item) {
        lock (Lock) {
            return List.Contains(item);
        }
    }
}
