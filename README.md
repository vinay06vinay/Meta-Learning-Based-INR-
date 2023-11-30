# Meta-Learning-Based-INR-
Meta Learning Based INR combined with Grid Based INR for CMSC828I final Project

If people are trying brute forcing and considering all parameters for meta learning using trans inr this are the places which probably need modifications:

# Code Modification Checklist

## 1. Update `hypo_shacira.py` in the `hyponets` folder:

- [ ] Add the forward pass function that calls one of the methods from the `Pipeline` class of WISP.

## 2. Modify `imgrec_trainer.py` in the `_iter_step()` function:

- [ ] Modify how the input for the forward pass is made.
- [ ] Call the `forward_pass` method of the `HypoShacira` class for the forward pass.

## 3. Update `trans_inr.py`:

- [ ] Modify the forward pass function for smooth weight assignment for the Shacira weights.
- [ ] Add the definition for the `set_params` method being called in the same code.


## 4. Update classes, especially in `hypo_shacira`:

- [ ] Add a for loop similar to `hypo_mlp.py` (in the same format) in `hypo_shacira` to assign parameter shapes.

## 5. Go to the `trans_inr.py` file:

- [ ] In the constructor, debug to ensure that the weight registration happens seamlessly.



# <span style="font-size: larger;">***__Note__***</span>

<span style="font-size: larger;"><strong><em><u>This is a very small and high level list</u></em></strong> If you guys get stuck around some point, add those tasks if you aren't free to do it.</span>

