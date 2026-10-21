# TPU Usage Guide

Use this guide to prepare your laptop, connect to an assigned TPU session, run your work and save checkpoints.

> **Keep your main copy on your own computer and back up frequently.** The VM is a temporary remote computer. Interruption, deletion or replacement can lose its local code, notebooks, data and running jobs. **Saving a file on the VM is not a backup.** Copy saved work to your own storage every **10–15 minutes**, after important changes/results, and before your session ends. Prefer developing on your own computer and deploying to the TPU VM when you need TPU computation. Closing SSH alone does not delete files, but it does not back them up either.

**Develop on your own computer**:

-  Keep the main copy of your source code, notebooks and important data on your own computer or storage you control. Develop and debug locally where possible; use the TPU VM just for TPU jobs.
-  Keep versioned code in your own repository and back up important data/results to your own external drive or cloud storage. Git alone does not back up uncommitted changes, ignored files or datasets.
-  Transfer only the code, dependencies and data needed for a run to the TPU VM. If you edit code or notebooks on the VM, copy those changes back frequently using step 3 before continuing local development.
-  Make sure your own storage has room for the backups. The course bucket is limited and is not your permanent project archive. You do not need Google Cloud credentials on your laptop to use this guide's SSH file-copy commands.


- **Your instructor provides:** a session, your Linux username, connection details, a prepared Python environment and a checkpoint bucket.
- **You provide:** your own SSH public key and its fingerprint. The key authorizes your Linux login; signing into a Google account does not authorize SSH.

## Get Started

### Create an SSH key on your laptop

-  Use the computer you will connect from.
-  Check whether the key path below already exists. Reuse it only if its private key is protected by a passphrase; otherwise choose a new filename and use it consistently throughout this guide.
- **Do not overwrite an existing key you still use.** If reusing a key, skip the key-generation command and run only the fingerprint command.
-  For a new key, run the commands for your operating system below.

**macOS/Linux — laptop terminal:**

```bash
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/cs239_tpu
ssh-keygen -lf ~/.ssh/cs239_tpu.pub -E sha256
```

**Windows with OpenSSH — laptop PowerShell:**

```powershell
New-Item -ItemType Directory -Force -Path "$HOME/.ssh" | Out-Null
ssh-keygen -t ed25519 -f "$HOME/.ssh/cs239_tpu"
ssh-keygen -lf "$HOME/.ssh/cs239_tpu.pub" -E sha256
```

**Check the result:**

- Private key: `~/.ssh/cs239_tpu`.
- Public key: `~/.ssh/cs239_tpu.pub`.
- Fingerprint: a `SHA256:...` value printed by the last command.

### Submit only your public key

-  Send **`~/.ssh/cs239_tpu.pub` and its fingerprint** through the instructor's approved roster process.
- **Keep `~/.ssh/cs239_tpu` private on your laptop.**
- **Use your own login.** Every student receives an individual username and key authorization.

**Which shell to use later:**

- For laptop commands in PowerShell, replace `~/.ssh/cs239_tpu` with `$HOME/.ssh/cs239_tpu`.
- After SSH connects, commands run in the TPU VM's Linux shell, including when your laptop uses Windows.

## Connect when your session is ready

### Check your instructor's handoff

-  Wait for the instructor to confirm the session is ready. An accepted cloud request may still be waiting for hardware.

**Before connecting, confirm you have:**

-  Assigned username and assigned host ip
-  Your team's bucket and checkpoint prefix.
-  Session end time, with its time zone.

### Connect from your laptop

-  Replace the uppercase placeholders with your assigned username and host, then run:

```bash
ssh -o IdentitiesOnly=yes -o ForwardAgent=no -i ~/.ssh/cs239_tpu ASSIGNED_USERNAME@ASSIGNED_HOST
```

-  **Compare the host-key fingerprint with the instructor's value before accepting it.**
-  Once connected, activate the supplied Python environment.

```bash
source ~/.venvs/cs239/bin/activate
```

**When installing dependencies:**

- Install project dependencies only inside your user environment.
- Ask the instructor for system-package changes. Student accounts do not have sudo or privileged container access.

## Check that JAX sees a TPU

-  **Inside the TPU VM, with the prepared Python environment active,** run:

```bash
python3 - <<'PY'
import jax
import jax.numpy as jnp

print(jax.devices())
assert jax.devices()[0].platform == 'tpu'
x = jnp.ones((32, 32))
y = (x @ x).block_until_ready()
assert float(y[0, 0]) == 32.0
print('TPU computation passed')
PY
```

-  Confirm that a TPU device is listed and the output includes **`TPU computation passed`**.
- **If JAX reports CPU only or cannot import:** stop and report the output to the instructor. Do not continue the session silently on CPU.


