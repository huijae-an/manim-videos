from manim import *
import numpy as np

class PCA(Scene):
    def construct(self):
        # ---------------------------
        # Part 1: PCA on 2D Data
        # ---------------------------
        
        # 1. Introduction to PCA
        title = Text("Principal Component Analysis (PCA)", font_size=36)
        subtitle = Text("Simplifying Data by Reducing Features", font_size=24).next_to(title, DOWN)
        self.play(Write(title), run_time=2)
        self.play(Write(subtitle), run_time=2)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=1)
        
        # 2. Setup 2D Coordinate System
        axes = Axes(
            x_range=[0, 15, 1],
            y_range=[0, 15, 1],
            x_length=8,
            y_length=8,
            axis_config={"color": WHITE}
        ).to_edge(LEFT, buff=1)
        
        labels = axes.get_axis_labels("Feature 1", "Feature 2")
        self.play(Create(axes), Write(labels), run_time=2)
        
        # 3. Plotting Data Points
        # Define three groups of points
        np.random.seed(42)  # For reproducibility
        group1 = np.random.multivariate_normal([5, 5], [[1, 0.5], [0.5, 1]], 30)
        group2 = np.random.multivariate_normal([10, 10], [[1, -0.3], [-0.3, 1]], 30)
        group3 = np.random.multivariate_normal([5, 10], [[1, 0], [0, 1]], 30)
        
        # Create Manim Dot objects
        dots_group1 = VGroup(*[Dot(axes.c2p(x, y), color=BLUE) for x, y in group1])
        dots_group2 = VGroup(*[Dot(axes.c2p(x, y), color=GREEN) for x, y in group2])
        dots_group3 = VGroup(*[Dot(axes.c2p(x, y), color=RED) for x, y in group3])
        
        # Animate dots
        self.play(FadeIn(dots_group1), run_time=1)
        self.play(FadeIn(dots_group2), run_time=1)
        self.play(FadeIn(dots_group3), run_time=1)
        
        # 4. Combining All Dots
        all_dots = VGroup(dots_group1, dots_group2, dots_group3)
        self.wait(1)
        
        # 5. Compute PCA
        # Combine all points
        all_points = np.vstack((group1, group2, group3))
        # Compute mean
        mean = np.mean(all_points, axis=0)
        # Center the data
        centered_data = all_points - mean
        # Compute covariance matrix
        cov_matrix = np.cov(centered_data.T)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 6. Plot Principal Components
        # Scaling factor for visualization
        scale = 5
        # First principal component
        pc1 = eigenvectors[:, 0] * scale * np.sqrt(eigenvalues[0])
        pc1_start = axes.c2p(mean[0], mean[1])
        pc1_end = axes.c2p(mean[0] + pc1[0], mean[1] + pc1[1])
        pc1_arrow = Arrow(start=pc1_start, end=pc1_end, buff=0, color=YELLOW, stroke_width=4)
        pc1_label = Text("PC1", font_size=18, color=YELLOW).next_to(pc1_end, RIGHT)
        
        # Second principal component
        pc2 = eigenvectors[:, 1] * scale * np.sqrt(eigenvalues[1])
        pc2_start = axes.c2p(mean[0], mean[1])
        pc2_end = axes.c2p(mean[0] + pc2[0], mean[1] + pc2[1])
        pc2_arrow = Arrow(start=pc2_start, end=pc2_end, buff=0, color=ORANGE, stroke_width=4)
        pc2_label = Text("PC2", font_size=18, color=ORANGE).next_to(pc2_end, RIGHT)
        
        # Animate principal components
        self.play(Create(pc1_arrow), Write(pc1_label), run_time=2)
        self.play(Create(pc2_arrow), Write(pc2_label), run_time=2)
        
        # 7. Highlight Orthogonality
        right_angle = RightAngle(pc1_arrow, pc2_arrow, length=0.3, color=WHITE)
        self.play(Create(right_angle), run_time=1)
        orthogonality_text = Text("Orthogonal Components", font_size=18, color=WHITE).next_to(right_angle, UP, buff=0.1)
        self.play(Write(orthogonality_text), run_time=1)
        
        # 8. Project Points onto Principal Components
        projection_lines_pc1 = VGroup()
        projected_dots_pc1 = VGroup()
        projection_lines_pc2 = VGroup()
        projected_dots_pc2 = VGroup()
        
        for point in all_points:
            # Projection onto PC1
            projection_length_pc1 = np.dot(point - mean, eigenvectors[:, 0])
            proj_point_pc1 = mean + projection_length_pc1 * eigenvectors[:, 0]
            proj_point_pc1_2d = axes.c2p(proj_point_pc1[0], proj_point_pc1[1])
            original_point_2d = axes.c2p(point[0], point[1])
            line_pc1 = DashedLine(start=original_point_2d, end=proj_point_pc1_2d, color=YELLOW)
            projection_lines_pc1.add(line_pc1)
            dot_pc1 = Dot(proj_point_pc1_2d, color=YELLOW)
            projected_dots_pc1.add(dot_pc1)
            
            # Projection onto PC2
            projection_length_pc2 = np.dot(point - mean, eigenvectors[:, 1])
            proj_point_pc2 = mean + projection_length_pc2 * eigenvectors[:, 1]
            proj_point_pc2_2d = axes.c2p(proj_point_pc2[0], proj_point_pc2[1])
            line_pc2 = DashedLine(start=original_point_2d, end=proj_point_pc2_2d, color=ORANGE)
            projection_lines_pc2.add(line_pc2)
            dot_pc2 = Dot(proj_point_pc2_2d, color=ORANGE)
            projected_dots_pc2.add(dot_pc2)
        
        # Animate projections
        self.play(Create(projection_lines_pc1), run_time=2)
        self.play(FadeIn(projected_dots_pc1), run_time=1)
        self.play(Create(projection_lines_pc2), run_time=2)
        self.play(FadeIn(projected_dots_pc2), run_time=1)
        
        # 9. Conclusion for 2D PCA
        summary_pca = Text("PCA identifies the directions of maximum variance.", font_size=24)
        self.play(
            FadeOut(all_dots), FadeOut(pc1_arrow), FadeOut(pc1_label),
            FadeOut(pc2_arrow), FadeOut(pc2_label),
            FadeOut(right_angle), FadeOut(orthogonality_text),
            FadeOut(projection_lines_pc1), FadeOut(projected_dots_pc1),
            FadeOut(projection_lines_pc2), FadeOut(projected_dots_pc2),
            Write(summary_pca), run_time=3
        )
        self.wait(2)
        
        # ---------------------------
        # Part 2: PCA on Image Data (Apple Example)
        # ---------------------------
        
        # 10. Introduction to Image PCA
        self.play(FadeOut(summary_pca), run_time=1)
        title_image = Text("PCA for Image Resolution Reduction", font_size=36)
        subtitle_image = Text("Compressing Images by Reducing Pixel Components", font_size=24).next_to(title_image, DOWN)
        self.play(Write(title_image), run_time=2)
        self.play(Write(subtitle_image), run_time=2)
        self.play(FadeOut(title_image), FadeOut(subtitle_image), run_time=1)
        
        # 11. Setup Axes for Image
        # Reuse the same axes, but reset them
        axes_image = Axes(
            x_range=[0, 15, 1],
            y_range=[0, 15, 1],
            x_length=8,
            y_length=8,
            axis_config={"color": WHITE}
        ).to_edge(LEFT, buff=1)
        labels_image = axes_image.get_axis_labels("X", "Y")
        self.play(Create(axes_image), Write(labels_image), run_time=2)
        
        # 12. Create Apple Representation with Dots
        # For simplicity, create an apple-like shape using parametric equations
        apple_points = []
        for theta in np.linspace(0, 2 * np.pi, 200):
            r = 5 + 3 * np.sin(theta)
            x = r * np.cos(theta) + 7  # Shifted to the right
            y = r * np.sin(theta) + 7  # Shifted upwards
            apple_points.append((x, y))
        
        # Add some irregularities to make it more apple-like
        for i in range(0, len(apple_points), 20):
            apple_points[i] = (apple_points[i][0] + np.random.uniform(-0.5, 0.5),
                               apple_points[i][1] + np.random.uniform(-0.5, 0.5))
        
        # Create Manim Dots for Apple
        apple_dots = VGroup(*[Dot(axes_image.c2p(x, y), color=RED, radius=0.02) for x, y in apple_points])
        self.play(FadeIn(apple_dots), run_time=2)
        
        # 13. Compute PCA on Image Pixels
        # Each pixel is a point (x, y)
        image_points = np.array(apple_points)
        # Compute mean
        mean_image = np.mean(image_points, axis=0)
        # Center the data
        centered_image = image_points - mean_image
        # Compute covariance matrix
        cov_image = np.cov(centered_image.T)
        # Compute eigenvalues and eigenvectors
        eigenvals_image, eigenvecs_image = np.linalg.eig(cov_image)
        # Sort eigenvectors by eigenvalues descending
        sorted_indices_image = np.argsort(eigenvals_image)[::-1]
        eigenvals_image = eigenvals_image[sorted_indices_image]
        eigenvecs_image = eigenvecs_image[:, sorted_indices_image]
        
        # 14. Plot Principal Components for Image
        # Scaling factor for visualization
        scale_image = 5
        # First principal component
        pc1_image = eigenvecs_image[:, 0] * scale_image * np.sqrt(eigenvals_image[0])
        pc1_image_start = axes_image.c2p(mean_image[0], mean_image[1])
        pc1_image_end = axes_image.c2p(mean_image[0] + pc1_image[0], mean_image[1] + pc1_image[1])
        pc1_image_arrow = Arrow(start=pc1_image_start, end=pc1_image_end, buff=0, color=YELLOW, stroke_width=4)
        pc1_image_label = Text("PC1", font_size=18, color=YELLOW).next_to(pc1_image_end, RIGHT)
        
        # Second principal component
        pc2_image = eigenvecs_image[:, 1] * scale_image * np.sqrt(eigenvals_image[1])
        pc2_image_start = axes_image.c2p(mean_image[0], mean_image[1])
        pc2_image_end = axes_image.c2p(mean_image[0] + pc2_image[0], mean_image[1] + pc2_image[1])
        pc2_image_arrow = Arrow(start=pc2_image_start, end=pc2_image_end, buff=0, color=ORANGE, stroke_width=4)
        pc2_image_label = Text("PC2", font_size=18, color=ORANGE).next_to(pc2_image_end, RIGHT)
        
        # Animate principal components for image
        self.play(Create(pc1_image_arrow), Write(pc1_image_label), run_time=2)
        self.play(Create(pc2_image_arrow), Write(pc2_image_label), run_time=2)
        
        # 15. Highlight Orthogonality for Image PCA
        right_angle_image = RightAngle(pc1_image_arrow, pc2_image_arrow, length=0.3, color=WHITE)
        self.play(Create(right_angle_image), run_time=1)
        orthogonality_text_image = Text("Orthogonal Components", font_size=18, color=WHITE).next_to(right_angle_image, UP, buff=0.1)
        self.play(Write(orthogonality_text_image), run_time=1)
        
        # 16. Project Image Pixels onto Principal Components
        projection_lines_image_pc1 = VGroup()
        projected_dots_image_pc1 = VGroup()
        projection_lines_image_pc2 = VGroup()
        projected_dots_image_pc2 = VGroup()
        
        for point in image_points:
            # Projection onto PC1
            projection_length_pca1 = np.dot(point - mean_image, eigenvecs_image[:, 0])
            proj_point_pca1 = mean_image + projection_length_pca1 * eigenvecs_image[:, 0]
            proj_point_pca1_2d = axes_image.c2p(proj_point_pca1[0], proj_point_pca1[1])
            original_point_2d_image = axes_image.c2p(point[0], point[1])
            line_pca1 = DashedLine(start=original_point_2d_image, end=proj_point_pca1_2d, color=YELLOW)
            projection_lines_image_pc1.add(line_pca1)
            dot_pca1 = Dot(proj_point_pca1_2d, color=YELLOW)
            projected_dots_image_pc1.add(dot_pca1)
            
            # Projection onto PC2
            projection_length_pca2 = np.dot(point - mean_image, eigenvecs_image[:, 1])
            proj_point_pca2 = mean_image + projection_length_pca2 * eigenvecs_image[:, 1]
            proj_point_pca2_2d = axes_image.c2p(proj_point_pca2[0], proj_point_pca2[1])
            line_pca2 = DashedLine(start=original_point_2d_image, end=proj_point_pca2_2d, color=ORANGE)
            projection_lines_image_pc2.add(line_pca2)
            dot_pca2 = Dot(proj_point_pca2_2d, color=ORANGE)
            projected_dots_image_pc2.add(dot_pca2)
        
        # Animate projections for image
        self.play(Create(projection_lines_image_pc1), run_time=2)
        self.play(FadeIn(projected_dots_image_pc1), run_time=1)
        self.play(Create(projection_lines_image_pc2), run_time=2)
        self.play(FadeIn(projected_dots_image_pc2), run_time=1)
        
        # 17. Reconstruct Image with Reduced Components
        # For demonstration, reconstruct using only PC1
        reconstructed_image_pc1 = mean_image + (projection_lines_image_pc1.get_all_points() - axes_image.c2p(*mean_image)).reshape(-1, 2) @ eigenvecs_image[:, 0].reshape(2, 1)
        reconstructed_image_pc1 = reconstructed_image_pc1 + mean_image
        reconstructed_image_pc1_dots = VGroup(*[
            Dot(axes_image.c2p(x, y), color=GREEN, radius=0.015)
            for x, y in reconstructed_image_pc1
        ])
        self.play(FadeOut(orthogonality_text_image), FadeOut(right_angle_image),
                  FadeOut(pc1_image_arrow), FadeOut(pc1_image_label),
                  FadeOut(pc2_image_arrow), FadeOut(pc2_image_label),
                  FadeOut(projection_lines_image_pc1), FadeOut(projected_dots_image_pc1),
                  FadeOut(projection_lines_image_pc2), FadeOut(projected_dots_image_pc2),
                  run_time=2)
        
        self.play(FadeIn(reconstructed_image_pc1_dots), run_time=2)
        reconstructed_label_pc1 = Text("Reconstructed with PC1", font_size=24, color=GREEN).to_edge(UP)
        self.play(Write(reconstructed_label_pc1), run_time=1)
        self.wait(2)
        
        # 18. Conclusion for Image PCA
        summary_image_pca = Text("PCA can reduce image resolution by retaining key components.", font_size=24)
        self.play(
            FadeOut(axes_image), FadeOut(labels_image), FadeOut(apple_dots),
            FadeOut(reconstructed_image_pc1_dots), FadeOut(reconstructed_label_pc1),
            FadeOut(summary_image_pca), run_time=1
        )
        self.wait(2)