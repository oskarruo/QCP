import torch


class TrainerTorch:
    def __init__(self, loss_fn, lr=0.01):
        self.loss_fn = loss_fn
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def train(self, n_iters, loss_kwargs):
        params = loss_kwargs["params"]
        params = torch.tensor(
            params,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam([params], lr=self.lr)

        self.losses = []

        fixed_kwargs = loss_kwargs.copy()
        fixed_kwargs.pop("params")

        for i in range(n_iters):
            optimizer.zero_grad()

            loss = self.loss_fn(params, **fixed_kwargs)
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())

            if (i + 1) % 10 == 0 or i == 0:
                print(f"Step {i + 1:03d} → Loss: {loss.item():.6f}")

        self.final_params = params.detach().cpu().numpy()
