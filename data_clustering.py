from manim import *
import numpy as np

class KMeansClusteringVisualization(Scene):
    def construct(self):
        # Title
        title = Text("K-Means Clustering", font_size=48)
        self.play(Write(title))
        self.wait(0.7)
        self.play(FadeOut(title))

        # Generate synthetic data
        np.random.seed(42)
        # Define parameters for three clusters
        cluster_1 = np.random.multivariate_normal(
            mean=[2, 2], cov=[[0.5, 0], [0, 0.5]], size=50)
        cluster_2 = np.random.multivariate_normal(
            mean=[8, 3], cov=[[0.5, 0], [0, 0.5]], size=50)
        cluster_3 = np.random.multivariate_normal(
            mean=[5, 8], cov=[[0.5, 0], [0, 0.5]], size=50)
        # Combine the clusters to form the dataset
        data = np.vstack((cluster_1, cluster_2, cluster_3))

        # Remove data points outside the desired range
        data = data[(data[:, 0] >= 0) & (data[:, 0] <= 10) & (data[:, 1] >= 0) & (data[:, 1] <= 10)]

        # Create a coordinate plane
        plane = NumberPlane(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=6,
            background_line_style={"stroke_color": BLUE_D, "stroke_width": 1},
            axis_config={"color": BLUE},
        )

        # Axis labels
        x_label = Text("X-axis", font_size=24).next_to(plane.x_axis.get_end(), DOWN)
        y_label = Text("Y-axis", font_size=24).next_to(plane.y_axis.get_top(), LEFT)

        self.play(Create(plane), Write(x_label), Write(y_label))

        # Plot initial data points in gray
        initial_scatter = VGroup()
        for point in data:
            initial_scatter.add(Dot(plane.c2p(point[0], point[1]), radius=0.04, color=GRAY))

        self.play(Create(initial_scatter))
        self.wait(0.7)

        # Explanation text
        explanation = Text("Initial Data Points", font_size=24)
        explanation.next_to(plane, UP)
        self.play(Write(explanation))
        self.wait(1)

        # Update explanation text
        new_explanation = Text("Data Points After Clustering", font_size=24)
        new_explanation.next_to(plane, UP)
        self.play(Transform(explanation, new_explanation))
        self.wait(0.5)

        # Number of clusters
        k = 3

        # Perform K-Means Clustering to get final centers
        # (Compute cluster centers without visualizing the initialization and movement)
        centers = data[np.random.choice(range(len(data)), size=k, replace=False)]
        max_iters = 100
        for iteration in range(max_iters):
            # Assign clusters
            distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update cluster centers
            new_centers = np.array([
                data[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                for i in range(k)
            ])

            # Check for convergence
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        # Visualize final cluster centers
        center_dots = VGroup()
        for center in centers:
            dot = Dot(plane.c2p(center[0], center[1]), radius=0.08, color=YELLOW)
            center_dots.add(dot)

        self.play(Create(center_dots))
        self.wait(0.7)

        # Assign colors to clusters
        colors = [RED, GREEN, BLUE]
        clustered_scatter = VGroup()
        for i, point in enumerate(data):
            cluster_idx = labels[i]
            color = colors[cluster_idx % len(colors)]
            clustered_scatter.add(Dot(plane.c2p(point[0], point[1]), radius=0.04, color=color))

        # Update the scatter plot with cluster colors
        self.play(Transform(initial_scatter, clustered_scatter))
        self.wait(0.5)

        # Add final cluster centers
        final_center_dots = VGroup()
        for center in centers:
            dot = Dot(plane.c2p(center[0], center[1]), radius=0.08, color=YELLOW)
            final_center_dots.add(dot)

        self.play(FadeIn(final_center_dots))
        # self.wait(0.5)

        # Draw lines from points to their cluster centers
        projection_lines = VGroup()
        for i, point in enumerate(data):
            cluster_idx = labels[i]
            center = centers[cluster_idx]
            line = Line(
                plane.c2p(point[0], point[1]),
                plane.c2p(center[0], center[1]),
                color=colors[cluster_idx % len(colors)],
                stroke_width=0.5
            )
            projection_lines.add(line)

        self.play(Create(projection_lines), run_time=2)
        self.wait(0.5)


        self.play(
            FadeOut(plane), FadeOut(initial_scatter), FadeOut(explanation),
            FadeOut(center_dots), FadeOut(final_center_dots), FadeOut(x_label), FadeOut(y_label)
        )
        self.wait(0.2)
        self.play(FadeOut(projection_lines))
        self.wait(0.5)

        # Centered summary text using Paragraph
        summary = Paragraph(
            "K-Means Clustering groups similar data points by",
            "iteratively updating cluster centers to minimize differences.",
            font_size=34,
            line_spacing=1.2,
            alignment="center"
        )

        # Center the entire paragraph on the screen
        summary.move_to(ORIGIN)  # Ensures the text block is truly centered

        self.play(Write(summary))
        self.wait(1.0)
        self.play(FadeOut(summary))