import dashscope
from dashscope.audio.asr import Recognition
import inspect

try:
    import importlib.metadata
    version = importlib.metadata.version("dashscope")
    print(f"✅ DashScope SDK Version: {version}")
except:
    print("Version unknown")

# Inspect the 'call' method
if hasattr(Recognition, 'call'):
    # Check if it is a function, method, staticmethod, classmethod
    attr = inspect.getattr_static(Recognition, 'call')
    print(f"🔍 Recognition.call type: {type(attr)}")
    
    # Check signature
    try:
        sig = inspect.signature(Recognition.call)
        print(f"📜 Signature: {sig}")
    except:
        print("Could not get signature")
else:
    print("❌ Recognition.call does NOT exist on the class directly.")

# Verify if we can call it without instantiation (Dry Run)
try:
    print("🚀 Attempting Class-level call (Recognition.call)...")
    # We pass dummy args to trigger signature check, not actual API
    Recognition.call(model='paraformer-v1', file='dummy.mp4')
except TypeError as e:
    print(f"❌ TypeError Caught: {e}")
    if "self" in str(e):
        print("   -> CONFIRMED: It requires instantiation (instance method).")
    else:
        print("   -> Signature error, but maybe not 'self' related.")
except Exception as e:
    print(f"⚠️ Other Error (Expected): {e}")

# Verify Instance call
try:
    print("🚀 Attempting Instance-level call (Recognition().call)...")
    rec = Recognition(model='paraformer-v1', file='dummy.mp4')
    if hasattr(rec, 'call'):
        rec.call()
    else:
        print("Instance has no call method?")
except Exception as e:
    print(f"⚠️ Instance Call Result: {e}")
